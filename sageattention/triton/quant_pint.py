import torch
import triton
import triton.language as tl

def bf16_to_fixed_point_torch(x_bf16: torch.Tensor):
    """
    将 [0, 1) 范围内的 bfloat16 tensor 转换为定点数 (Scale = 1/2^16)
    返回: 定点数tensor (模拟 uint32 存储) 和 scale
    """
    # 确保输入是 bfloat16 类型
    if x_bf16.dtype != torch.bfloat16:
        print(f"警告: 输入 dtype 为 {x_bf16.dtype}，已转换为 bfloat16")
        x_bf16 = x_bf16.to(torch.bfloat16)

    # 1. 提取 bf16 的底层位模式
    # PyTorch 无法直接对浮点数做位运算，需 view 成 int16。
    # 转为 int32 并按位与 0xFFFF 是为了避免负数符号扩展带来的干扰。
    x_int = x_bf16.view(torch.int16).to(torch.int32) & 0xFFFF

    # 提取8位 exponent (去掉符号位，[0,1)的数据符号位为0)
    # exponent 位于第 [7:14] 位
    raw_exponent = (x_int >> 7) & 0xFF

    # 提取7位 mantissa
    mantissa = x_int & 0x7F
    # del x_int  # 不再需要 x_int

    # mantissa 加上默认的 1 (即第7位置为1)
    m_int = mantissa | 0x80
    # del mantissa  # 不再需要 mantissa

    # 还原真实 exponent (减去 bias 127)
    exponent = raw_exponent - 127
    # del raw_exponent  # 不再需要 raw_exponent
    
    # 使用 int32 模拟 uint32 存储
    exponent_tensor = exponent.to(torch.int32)
    m_int_tensor = m_int.to(torch.int32)
    # del exponent, m_int  # 不再需要 exponent 和 m_int

    # 2. 定点化移位操作
    # 基础移位量
    shift = exponent_tensor + 9

    # 初始化输出 tensor 为 0 (这也自动处理了 exponent < -15 等"其他"情况)
    out_fixed = torch.zeros_like(m_int_tensor, dtype=torch.int32)

    # 划分掩码条件
    mask_left  = (exponent_tensor >= -9) & (exponent_tensor <= -1)
    mask_right = (exponent_tensor >= -15) & (exponent_tensor <= -10)
    # del exponent_tensor  # 不再需要 exponent_tensor

    # 对 exponent 在 [-9, -1] 范围的数进行左移
    out_fixed[mask_left] = m_int_tensor[mask_left] << shift[mask_left]

    # 对 exponent 在 [-15, -10] 范围的数进行右移（注意 shift 为负数，加负号转为正的右移量）
    # 右移会自动舍去低位 mantissa
    out_fixed[mask_right] = m_int_tensor[mask_right] >> (-shift[mask_right])
    # del m_int_tensor, shift, mask_left, mask_right  # 不再需要这些中间 tensor

    # 3. 输出定点数与 scale
    scale = 1 / (2 ** 16)

    return out_fixed, scale


# ==========================================
# Triton Kernel 实现 (已修复)
# ==========================================
@triton.jit
def bf16_to_fixed_point(x):
    # 1. 提取 bf16 的底层位模式
    # 【修复2】使用 .to(bitcast=True) 转为 int16，再向上转换为 int32 并用掩码截断
    # 这是跨 Triton 版本最稳健的比特级别提取方式
    x_int16 = x.to(tl.int16, bitcast=True)
    x_int32 = x_int16.to(tl.int32) & 0xFFFF

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

    # 安全地计算移位量
    shift_left_amt = tl.where(mask_left, shift, 0)
    shift_right_amt = tl.where(mask_right, -shift, 0)

    # 执行移位
    val_left = m_int << shift_left_amt
    val_right = m_int >> shift_right_amt

    # 初始化输出为 0，并应用掩码选择正确的分支
    out = tl.zeros_like(x_int32)
    out = tl.where(mask_left, val_left, out)
    out = tl.where(mask_right, val_right, out)
    
    # 【修复1】必须显式 return，Triton 内部函数不能依赖形参进行 inplace 修改
    return out

@triton.jit
def bf16_to_fixed_point_s8(x):
    # 1. 提取 bf16 的底层位模式
    # 【修复2】使用 .to(bitcast=True) 转为 int16，再向上转换为 int32 并用掩码截断
    # 这是跨 Triton 版本最稳健的比特级别提取方式
    x_int16 = x.to(tl.int16, bitcast=True)
    x_int32 = x_int16.to(tl.int32) & 0xFFFF

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

    # 安全地计算移位量
    shift_left_amt = tl.where(mask_left, shift, 0)
    shift_right_amt = tl.where(mask_right, -shift, 0)

    # 执行移位
    val_left = m_int << shift_left_amt
    val_right = m_int >> shift_right_amt

    # 初始化输出为 0，并应用掩码选择正确的分支
    out = tl.zeros_like(x_int32)
    out = tl.where(mask_left, val_left, out)
    out = tl.where(mask_right, val_right, out)
    out_hi =( out >> 8) & 0xFF
    out_lo = out & 0xFF
    out_hi_s8 = (out_hi - 128).to(tl.int8)
    out_lo_s8 = (out_lo - 128).to(tl.int8)
    
    return out_hi_s8, out_lo_s8


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

    # 调用定点化函数 (接收返回值)
    out = bf16_to_fixed_point(x)
    
    # 3. 写回内存
    tl.store(out_ptr + offsets, out, mask=mask)

def bf16_to_fixed_point_triton(x_bf16: torch.Tensor):
    """
    Triton Kernel 的 Python 封装
    """
    assert x_bf16.is_cuda and x_bf16.is_contiguous(), "输入 Tensor 必须在 GPU 上且内存连续"
    if x_bf16.dtype != torch.bfloat16:
        print("输入 Tensor 类型不是 bfloat16，已转换为 bfloat16")
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
