import torch

def bf16_to_fixed_point(x_bf16: torch.Tensor):
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

    # mantissa 加上默认的 1 (即第7位置为1)
    m_int = mantissa | 0x80

    # 还原真实 exponent (减去 bias 127)
    exponent = raw_exponent - 127
    
    # 使用 int32 模拟 uint32 存储
    exponent_tensor = exponent.to(torch.int32)
    m_int_tensor = m_int.to(torch.int32)

    # 2. 定点化移位操作
    # 基础移位量
    shift = exponent_tensor + 9

    # 初始化输出 tensor 为 0 (这也自动处理了 exponent < -15 等"其他"情况)
    out_fixed = torch.zeros_like(m_int_tensor, dtype=torch.int32)

    # 划分掩码条件
    mask_left  = (exponent_tensor >= -9) & (exponent_tensor <= -1)
    mask_right = (exponent_tensor >= -15) & (exponent_tensor <= -10)

    # 对 exponent 在 [-9, -1] 范围的数进行左移
    out_fixed[mask_left] = m_int_tensor[mask_left] << shift[mask_left]

    # 对 exponent 在 [-15, -10] 范围的数进行右移（注意 shift 为负数，加负号转为正的右移量）
    # 右移会自动舍去低位 mantissa
    out_fixed[mask_right] = m_int_tensor[mask_right] >> (-shift[mask_right])

    # 3. 输出定点数与 scale
    scale = 1 / (2 ** 16)

    return out_fixed, scale

# ==========================================
# 验证测试代码
# ==========================================
if __name__ == "__main__":
    # 构造测试数据，包括正常小数，需要截断的极小小数，以及0
    x_test = torch.tensor([0.5, 0.25, 2**(-15), 2**(-16), 0.0], dtype=torch.bfloat16)
    
    fixed_tensor, scale = bf16_to_fixed_point(x_test)
    dequantized_x = fixed_tensor * scale

    print(f"原始 bf16 输入: {x_test.tolist()}")
    print(f"定点数存储结果: {fixed_tensor.tolist()} (dtype: {fixed_tensor.dtype})")
    print(f"反量化恢复数值: {dequantized_x.tolist()}")