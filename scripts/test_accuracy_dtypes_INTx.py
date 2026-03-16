import torch
import torchmm
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

def simulate_quantization(x: torch.Tensor, precision: str) -> torch.Tensor:
    """根据指定的精度对张量进行量化模拟"""
    precision = precision.upper()
    if precision == 'INT16':
        data, scale = dynamic_quantize_int(x, bits=16)
        return data.to(torch.int32), scale
    elif precision == 'INT8':
        data, scale = dynamic_quantize_int(x, bits=8)
        return data.to(torch.int32), scale
    elif precision == 'INT4':
        data, scale = dynamic_quantize_int(x, bits=4)
        return data.to(torch.int32), scale
    elif precision == 'INT2':
        data, scale = dynamic_quantize_int(x, bits=2)
        return data.to(torch.int32), scale
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
                                    qk_precision: str, p_precision: str, v_precision: str):
    """执行评估流程"""
    print(f"--- 评估配置: QK={qk_precision}, P={p_precision}, V={v_precision} ---")
    
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
    
    # S_q = Q_q @ K_q.transpose(-2, -1) * Q_scale * K_scale / math.sqrt(head_dim)
    # S_q = torch.zeros((Q_q.size(0), Q_q.size(1), K_q.size(1)), device=Q_q.device)
    if qk_precision == "INT4" or qk_precision == "INT8":
        S_q = torchmm.matmul(Q_q, K_q.transpose(-2, -1)) * Q_scale * K_scale / math.sqrt(head_dim)
    else:
        raise ValueError(f"不支持的 QK 精度类型: {qk_precision}")
    P_q = F.softmax(S_q, dim=-1)
    del Q_q, K_q
    # import ipdb; ipdb.set_trace()
    # torch.save(P_q, "P_q_fp16.pt")
    # P_q[P_q < 2**(-5)] = 0
    # P_q[P_q < 2**(-10)] = 0
    # P_q[P_q < 2**(-13)] = 0
    # P_q[P_q < 2**(-14)] = 0
    # P_q[P_q < 2**(-15)] = 0
    # P_q[P_q < 2**(-20)] = 0
    # P_q[P_q < 2**(-30)] = 0
    # P_q[P_q < 2**(-40)] = 0
    
    # PV 阶段量化 (P是注意力权重，V是Value)
    P_qq, P_scale = simulate_quantization(P_q, p_precision)
    V_q, V_scale = simulate_quantization(V_ref, v_precision)
    # O_q = P_qq @ V_q
    if p_precision == "INT8" or p_precision == "INT16":
        O_q = torchmm.matmul(P_qq, V_q) * P_scale * V_scale  # 还原缩放
    else:
        raise ValueError(f"不支持的 P 精度类型: {p_precision}")
    
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
            {"qk": "INT8", "p": "INT16", "v": "INT16"},
            {"qk": "INT8", "p": "INT16", "v": "INT8"},
            {"qk": "INT8", "p": "INT16", "v": "INT4"},
            {"qk": "INT8", "p": "INT16", "v": "INT2"},
            # {"qk": "INT8", "p": "INT8", "v": "INT16"},
            # {"qk": "INT8", "p": "INT8", "v": "INT8"},
            {"qk": "INT4", "p": "INT16", "v": "INT16"},
            {"qk": "INT4", "p": "INT16", "v": "INT8"},
            {"qk": "INT4", "p": "INT16", "v": "INT4"},
            {"qk": "INT4", "p": "INT16", "v": "INT2"},
            # {"qk": "INT4", "p": "INT8", "v": "INT16"},
            # {"qk": "INT4", "p": "INT8", "v": "INT8"},
        ]
        
        with torch.no_grad():
            for cfg in configs:
                evaluate_attention_quantization(Q, K, V, qk_precision=cfg["qk"], p_precision=cfg["p"], v_precision=cfg["v"])
