import torch
import torch.nn.functional as F
import math
import os

def dynamic_quantize_int(x: torch.Tensor, bits: int) -> torch.Tensor:
    """动态对称整数(INT4/INT8)量化"""
    max_val = (1 << (bits - 1)) - 1
    # 动态计算 per-tensor scale
    scale = x.abs().max() / max_val
    if scale == 0:
        return x
    
    # 量化 -> 截断 -> 反量化
    x_q = torch.clamp(torch.round(x / scale), -max_val, max_val)
    # x_dq = x_q * scale
    return x_q, scale

def dynamic_quantize_fp8(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """动态 FP8 (E4M3/E5M2) 量化"""
    # E4M3 最大正常值为 448.0, E5M2 为 57344.0
    if dtype == torch.float8_e4m3fn:
        max_val = 448.0
    elif dtype == torch.float8_e5m2:
        max_val = 57344.0
    else:
        raise ValueError("不支持的 FP8 格式")

    scale = x.abs().max() / max_val
    if scale == 0:
        return x
    
    # 缩放 -> 转换至 FP8 截断精度 -> 转换回 FP32 并还原缩放
    x_scaled = x / scale
    x_fp8 = x_scaled.to(dtype)
    # scale_tensor = torch.tensor([scale], dtype=torch.float32, device=x.device)
    # x_dq = x_fp8.to(torch.float32) * scale
    return x_fp8, scale # scale_tensor

def simulate_quantization(x: torch.Tensor, precision: str) -> torch.Tensor:
    """根据指定的精度对张量进行量化模拟"""
    precision = precision.upper()
    if precision == 'FP32':
        return x.to(torch.float32), 1.0
    elif precision == 'FP16':
        return x.to(torch.float16), 1.0
    elif precision == 'BF16':
        return x.to(torch.bfloat16), 1.0
    elif precision == 'INT8':
        return dynamic_quantize_int(x, bits=8)
    elif precision == 'INT4':
        return dynamic_quantize_int(x, bits=4)
    elif precision == 'FP8_E4M3':
        if not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError("当前 PyTorch 版本过低，不支持原生的 FP8 数据类型，请升级至 >= 2.1")
        return dynamic_quantize_fp8(x, torch.float8_e4m3fn)
    elif precision == 'FP8_E5M2':
        if not hasattr(torch, 'float8_e5m2'):
            raise RuntimeError("当前 PyTorch 版本过低，不支持原生的 FP8 数据类型，请升级至 >= 2.1")
        return dynamic_quantize_fp8(x, torch.float8_e5m2)
    else:
        raise ValueError(f"不支持的精度类型: {precision}")

def calculate_metrics(ref_tensor: torch.Tensor, q_tensor: torch.Tensor) -> dict:
    """计算精度损失指标"""
    ref_flat = ref_tensor.flatten()
    q_flat = q_tensor.flatten()
    
    # 1. Cosine Similarity (余弦相似度)
    cos_sim = F.cosine_similarity(ref_flat, q_flat, dim=0).item()
    
    # 2. Relative L1 Error (相对 L1 误差)
    mean_abs_ref = torch.abs(ref_flat).mean()
    if mean_abs_ref == 0:
        rel_l1 = 0.0
    else:
        rel_l1 = (torch.abs(ref_flat - q_flat).mean() / mean_abs_ref).item()
        
    # 3. RMSE (均方根误差)
    rmse = torch.sqrt(torch.mean((ref_flat - q_flat) ** 2)).item()
    
    return {
        "Cosine Similarity": cos_sim,
        "Relative L1": rel_l1,
        "RMSE": rmse
    }

def evaluate_attention_quantization(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                    qk_precision: str, pv_precision: str):
    """执行评估流程"""
    print(f"--- 评估配置: QK={qk_precision}, PV={pv_precision} ---")
    
    # 确保基准输入为 FP16
    Q_ref, K_ref, V_ref = Q, K, V
    head_dim = Q_ref.size(-1)
    
    # ================= 1. 基准 FP32 前向传播 =================
    S_ref = Q_ref @ K_ref.transpose(-2, -1) / math.sqrt(head_dim)
    P_ref = F.softmax(S_ref, dim=-1)
    O_ref = P_ref @ V_ref
    del S_ref, P_ref
    
    # ================= 2. 量化模拟前向传播 =================
    # QK 阶段量化
    Q_q, Q_scale = simulate_quantization(Q_ref, qk_precision)
    K_q, K_scale = simulate_quantization(K_ref, qk_precision)
    
    S_q = Q_q @ K_q.transpose(-2, -1) * Q_scale * K_scale / math.sqrt(head_dim)
    P_q = F.softmax(S_q, dim=-1)
    del Q_q, K_q
    import ipdb; ipdb.set_trace()
    # torch.save(P_q, "P_q_fp16.pt")
    
    # PV 阶段量化 (P是注意力权重，V是Value)
    P_qq, P_scale = simulate_quantization(P_q, pv_precision)
    V_q, V_scale = simulate_quantization(V_ref, pv_precision)
    # O_q = P_qq @ V_q
    if pv_precision == "FP16" or pv_precision == "BF16" or pv_precision == "INT8":
        O_q = torch.bmm(P_qq, V_q) * P_scale * V_scale  # 还原缩放
    else:
        O_q = torch.bmm(P_qq.to(torch.float32), V_q.to(torch.float32)) * P_scale * V_scale  # 还原缩放
        O_q = O_q.to(torch.float16)
        # O_q = torch.nn.functional.scaled_mm(
        #               P_qq,
        #               V_q,
        #               scale_a=P_scale,
        #               scale_b=V_scale,
        #               out_dtype=torch.float16 # 指定输出类型为 FP16
        #               )
    
    # ================= 3. 评估指标计算 =================
    metrics = calculate_metrics(O_ref.to(torch.float32), O_q.to(torch.float32))
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("\n")
    return metrics

def load_or_generate_data(pt_path: str = None, batch_size=2, seq_len=128, head_dim=64):
    """处理输入：读取 .pt 文件或随机生成"""
    if pt_path and os.path.exists(pt_path):
        print(f"从 {pt_path} 加载 QKV 张量...")
        data = torch.load(pt_path)
        Q = data['query'].cuda().squeeze(0)
        K = data['key'].cuda().squeeze(0)
        V = data['value'].cuda().squeeze(0)
        print(f"成功加载 QKV 张量，形状为: Q={Q.shape}, K={K.shape}, V={V.shape}")
    else:
        print("未提供有效的 .pt 文件，生成随机正态分布张量作为测试数据...")
        Q = torch.randn(batch_size, seq_len, head_dim).cuda()
        K = torch.randn(batch_size, seq_len, head_dim).cuda()
        V = torch.randn(batch_size, seq_len, head_dim).cuda()
    return Q, K, V

if __name__ == "__main__":
    # 批量处理
    tensor_dir = '../'
    TEST_ALL = False
    if os.path.exists(tensor_dir) and TEST_ALL:
        tensor_paths = [os.path.join(tensor_dir, f) for f in os.listdir(tensor_dir) if f.endswith('.pt')]
    else:
        tensor_paths = ['../qkv_tensors_30.pt']  # fallback

    for PT_FILE_PATH in tensor_paths:
        # 加载数据
        # Q, K, V = load_or_generate_data(pt_path=None) # 将 None 换成 PT_FILE_PATH 即可读取文件
        Q, K, V = load_or_generate_data(pt_path=PT_FILE_PATH) # 将 None 换成 PT_FILE_PATH 即可读取文件
        
        # 测试不同的精度组合配置
        configs = [
            {"qk": "INT8", "pv": "FP16"},
            {"qk": "INT8", "pv": "BF16"},
            # {"qk": "INT8", "pv": "FP8_E4M3"},
            # {"qk": "INT8", "pv": "FP8_E5M2"},
            # {"qk": "INT8", "pv": "INT8"},
            {"qk": "INT4", "pv": "FP16"},
            {"qk": "INT4", "pv": "BF16"},
            # {"qk": "INT4", "pv": "FP8_E4M3"},
            # {"qk": "INT4", "pv": "FP8_E5M2"},
            # {"qk": "INT4", "pv": "INT8"},
        ]
        
        with torch.no_grad():
            for cfg in configs:
                evaluate_attention_quantization(Q, K, V, qk_precision=cfg["qk"], pv_precision=cfg["pv"])