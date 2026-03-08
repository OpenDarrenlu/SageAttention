import torch
import matplotlib.pyplot as plt
import argparse
import os

def get_bfloat16_exponent(tensor: torch.Tensor, unbiased=True):
    """
    提取 bf16 tensor 的指数部分
    :param tensor: 输入的 bfloat16 tensor
    :param unbiased: 是否返回减去偏移量(127)后的真实指数值
    """
    # 确保输入是 bfloat16
    assert tensor.dtype == torch.bfloat16, "Tensor must be of dtype bfloat16"
    
    # 1. 内存共享视角转换：将 bf16 当作 int16 处理，这一步是零拷贝的
    int_view = tensor.view(torch.int16)
    
    # 2. 右移 7 位 (跨过 7 bits 的 mantissa)
    # 3. 使用按位与 (& 0xFF) 截取最低的 8 bits (屏蔽掉可能因负数符号位右移带来的 1)
    stored_exponent = (int_view >> 7) & 0xFF
    
    if not unbiased:
        return stored_exponent

    # 4. 计算真实的指数值 (bfloat16 的偏移量与 FP32 一样，都是 127)
    # 注意：这里需要转成 int32，否则减 127 可能会导致 int16 溢出或越界
    actual_exponent = stored_exponent.to(torch.int32) - 127
    
    # 可选处理：如果 stored_exponent 是 0，代表数值是 0 或非规格化数
    # 如果 stored_exponent 是 255，代表 NaN 或 Inf
    
    return actual_exponent
  
def main(pt_file, bins=100, output_img="histogram.png"):
    # 1. 检查并分配设备 (优先使用 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用计算设备: {device}")

    # 2. 读取 .pt 文件并直接映射到计算设备上
    print(f"[*] 正在加载文件: {pt_file} ...")
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"找不到文件: {pt_file}")
        
    tensor = torch.load(pt_file, map_location=device)

    # 处理 .pt 是 state_dict (字典) 的情况
    if isinstance(tensor, dict):
        print("[!] 警告: 读取到的是一个字典 (可能是模型 state_dict)。")
        print("[!] 正在将字典中所有的 Tensor 展平并拼接在一起进行统计...")
        # 将所有张量展平并拼接在一起
        tensor = torch.cat([v.flatten() for v in tensor.values() if isinstance(v, torch.Tensor)])
    elif isinstance(tensor, torch.Tensor):
        tensor = tensor.flatten()
    else:
        raise TypeError(f"不支持的数据类型: {type(tensor)}")

    print(f"[*] 数据总数: {tensor.numel()} 个元素")
    # import ipdb; ipdb.set_trace()
    tensor_exponent = get_bfloat16_exponent(tensor)
    # 计算1. 大于0， 2. 等于0，3. [-8，0) 4. 等于-9，5. [-20, -10], 6. [-40,-20) ,7. 小于-40的数量比例
    greater_than_0 = (tensor_exponent > 0).sum().item()
    equal_to_0 = (tensor_exponent == 0).sum().item()
    between_minus_9_and_0 = ((tensor_exponent >= -9) & (tensor_exponent < 0)).sum().item()
    between_minus_20_and_minus_10 = ((tensor_exponent >= -20) & (tensor_exponent <= -10)).sum().item()
    between_minus_40_and_minus_20 = ((tensor_exponent >= -40) & (tensor_exponent < -20)).sum().item()
    less_than_minus_40 = (tensor_exponent < -40).sum().item()
    
    print(f"[*] 大于0的指数数量比例: {greater_than_0 / tensor.numel():.4f}")
    print(f"[*] 等于0的指数数量比例: {equal_to_0 / tensor.numel():.4f}")
    print(f"[*] [-9, 0) 范围内的指数数量比例: {between_minus_9_and_0 / tensor.numel():.4f}")
    print(f"[*] [-20, -10] 范围内的指数数量比例: {between_minus_20_and_minus_10 / tensor.numel():.4f}")
    print(f"[*] [-40, -20) 范围内的指数数量比例: {between_minus_40_and_minus_20 / tensor.numel():.4f}")
    print(f"[*] 小于-40的指数数量比例: {less_than_minus_40 / tensor.numel():.4f}")
    
    # # 2. 将 Tensor 转换为 Numpy 数组方便绘图
    # np_data = tensor_exponent.cpu().numpy()

    # # 3. 设置绘图
    # plt.figure(figsize=(10, 6))

    # # bins 设置为 128，确保每一个整数值都能落在对应的柱状条上
    # plt.hist(np_data, bins=128, range=(-127, 0), color='skyblue', edgecolor='black', alpha=0.7)

    # # 4. 添加图表修饰
    # plt.title('Tensor Exponent Distribution (-127 to 0)', fontsize=15)
    # plt.xlabel('Exponent Value', fontsize=12)
    # plt.ylabel('Frequency', fontsize=12)
    # plt.grid(axis='y', linestyle='--', alpha=0.6)

    # # 保存图片
    # plt.savefig(output_img, dpi=300, bbox_inches='tight')
    # print(f"[+] 绘制完成！图表已保存至: {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取 .pt 文件并利用 GPU 绘制数据分布直方图")
    parser.add_argument("pt_file", type=str, help="输入的 .pt 文件路径")
    parser.add_argument("--bins", type=int, default=150, help="直方图的柱子数量 (默认: 150)")
    parser.add_argument("--output", type=str, default="histogram_exponent.png", help="输出的图片路径 (默认: histogram.png)")
    
    args = parser.parse_args()
    main(args.pt_file, args.bins, args.output)