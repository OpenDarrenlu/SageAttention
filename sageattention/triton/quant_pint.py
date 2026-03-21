import torch
import triton
import triton.language as tl
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
