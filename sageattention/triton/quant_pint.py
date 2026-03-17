import torch
import triton
import triton.language as tl

# ==========================================
# Triton Kernel 实现
# ==========================================
@triton.jit
def bf16_to_fixed_point(x):
    # 1. 提取 bf16 的底层位模式
    # 直接 bitcast 为 uint16，避免转成有符号 int16 带来的符号扩展问题
    x_uint16 = tl.cast(x, tl.uint16, bitcast=True)
    x_int32 = x_uint16.to(tl.int32)

    # 提取8位 exponent 和 7位 mantissa
    raw_exponent = (x_int32 >> 7) & 0xFF
    mantissa = x_int32 & 0x7F
    
    # 加上默认的 1
    m_int = mantissa | 0x80
    
    # 还原真实 exponent
    exponent = raw_exponent - 127
    
    # 2. 定点化移位操作
    shift = exponent + 9

    # 划分掩码条件
    mask_left = (exponent >= -9) & (exponent <= -1)
    mask_right = (exponent >= -15) & (exponent <= -10)

    # 安全地计算移位量（避免 CUDA 中的负数移位未定义行为）
    shift_left_amt = tl.where(mask_left, shift, 0)
    shift_right_amt = tl.where(mask_right, -shift, 0)

    # 执行移位
    val_left = m_int << shift_left_amt
    val_right = m_int >> shift_right_amt

    # 初始化输出为 0，并应用掩码选择正确的分支
    out = tl.zeros_like(x_int32)
    out = tl.where(mask_left, val_left, out)
    out = tl.where(mask_right, val_right, out)

@triton.jit
def bf16_to_fixed_point_kernel(
    x_ptr,          # 输入 bfloat16 指针
    out_ptr,        # 输出 int32 指针
    n_elements,     # 元素总数
    BLOCK_SIZE: tl.constexpr, # 线程块大小
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载 bf16 数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 调用定点化函数
    out = bf16_to_fixed_point(x)
    
    # 3. 写回内存
    tl.store(out_ptr + offsets, out, mask=mask)


def bf16_to_fixed_point_triton(x_bf16: torch.Tensor):
    """
    Triton Kernel 的 Python 封装
    """
    assert x_bf16.is_cuda and x_bf16.is_contiguous(), "输入 Tensor 必须在 GPU 上且内存连续"
    if x_bf16.dtype != torch.bfloat16:
        x_bf16 = x_bf16.to(torch.bfloat16)

    n_elements = x_bf16.numel()
    
    # 分配输出内存
    out_fixed = torch.empty_like(x_bf16, dtype=torch.int32)
    
    # 自动调优的线程块大小，通常 1024 比较合适
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    bf16_to_fixed_point_kernel[grid](
        x_bf16, out_fixed, n_elements, 
        BLOCK_SIZE=1024
    )
    
    scale = 1 / (2 ** 16)
    return out_fixed, scale