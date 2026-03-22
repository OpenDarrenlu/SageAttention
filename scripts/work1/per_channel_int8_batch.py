import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
import time

# ==========================================
# 1. PyTorch 修正版基准 (严格遵循你的数学逻辑)
# ==========================================
def per_channel_int8_pytorch(v: torch.Tensor, tensor_layout: str = "BHND", smooth_v: bool = True):
    orig_shape = v.shape
    head_dim = orig_shape[-1]
    v_2d = v.reshape(-1, head_dim).float()
    mean = None
    if smooth_v:
        mean = v_2d.mean(dim=0)
        v_2d = v_2d - mean[None, :]
    
    abs_max = torch.max(torch.abs(v_2d), dim=0)[0]
    scale = abs_max / 127.0
    scale = torch.clamp(scale, min=1e-9)
    
    v_quant = torch.round(v_2d / scale[None, :])
    v_quant += 0.5 * torch.where(v_quant >= 0, 1, -1)
    return v_quant.to(torch.int8).reshape(orig_shape), scale, mean

# ==========================================
# 2. Triton Kernels
# ==========================================
@triton.jit
def _get_stats_kernel(
    V_ptr, Scale_ptr, Mean_ptr,
    M, D,
    stride_vm, stride_vd,
    SMOOTH_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    if SMOOTH_V:
        # --- Pass 1: 计算 Mean ---
        sum_vals = tl.zeros([BLOCK_D], dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_offsets = m + tl.arange(0, BLOCK_M)
            mask = (m_offsets < M)[:, None] & d_mask[None, :]
            v_ptrs = V_ptr + m_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)
            sum_vals += tl.sum(v, axis=0)
        
        mean = sum_vals / M
        
        # --- Pass 2: 中心化并计算 Max(Abs) ---
        max_abs = tl.zeros([BLOCK_D], dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_offsets = m + tl.arange(0, BLOCK_M)
            mask = (m_offsets < M)[:, None] & d_mask[None, :]
            v_ptrs = V_ptr + m_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)
            
            # 中心化并消除 mask 外的数据干扰
            v_centered = v - mean[None, :]
            v_centered = tl.where(mask, v_centered, 0.0)
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(v_centered), axis=0))
            
        scale = max_abs / 127.0
    else:
        # --- 只需要 1 个 Pass 计算 Max(Abs) ---
        max_abs = tl.zeros([BLOCK_D], dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_offsets = m + tl.arange(0, BLOCK_M)
            mask = (m_offsets < M)[:, None] & d_mask[None, :]
            v_ptrs = V_ptr + m_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)
            
            v = tl.where(mask, v, 0.0)
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(v), axis=0))
            
        scale = max_abs / 127.0
        mean = tl.zeros([BLOCK_D], dtype=tl.float32)

    # 存储结果
    scale = tl.maximum(scale, 1e-9)
    tl.store(Scale_ptr + d_offsets, scale, mask=d_mask)
    if SMOOTH_V:
        tl.store(Mean_ptr + d_offsets, mean, mask=d_mask)

@triton.jit
def _apply_quant_kernel(
    V_ptr, Quant_ptr, Scale_ptr, Mean_ptr,
    M, D,
    stride_vm, stride_vd,
    stride_qm, stride_qd,
    SMOOTH_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    
    mask = (m_offsets < M)[:, None] & (d_offsets < D)[None, :]
    
    v_ptrs = V_ptr + m_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vd
    q_ptrs = Quant_ptr + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    
    v = tl.load(v_ptrs, mask=mask).to(tl.float32)
    scale = tl.load(Scale_ptr + d_offsets, mask=d_offsets < D)
    
    if SMOOTH_V:
        mean = tl.load(Mean_ptr + d_offsets, mask=d_offsets < D)
        v = v - mean[None, :]
        
    q = v / scale[None, :]
    
    # # 使用 libdevice.rint 严格对齐 torch.round 的 "Round half to even" (银行家舍入法)
    # q = libdevice.rint(q)
    q += 0.5 * tl.where(q >= 0, 1, -1)
    
    tl.store(q_ptrs, q.to(tl.int8), mask=mask)

def per_channel_int8_triton(v: torch.Tensor, tensor_layout: str = "BHND", smooth_v: bool = True):
    v = v.contiguous()
    head_dim = v.shape[-1]
    M = v.numel() // head_dim
    D = head_dim
    
    v_2d = v.view(M, D)
    
    scale = torch.empty((D,), dtype=torch.float32, device=v.device)
    mean = torch.empty((D,), dtype=torch.float32, device=v.device) if smooth_v else None
    v_quant = torch.empty_like(v_2d, dtype=torch.int8)
    
    BLOCK_M_1 = 1024
    BLOCK_D_1 = triton.next_power_of_2(D) if D <= 64 else 64
    grid_1 = (triton.cdiv(D, BLOCK_D_1),)
    
    dummy_mean = mean if smooth_v else scale
    
    _get_stats_kernel[grid_1](
        v_2d, scale, dummy_mean,
        M, D,
        v_2d.stride(0), v_2d.stride(1),
        SMOOTH_V=smooth_v,
        BLOCK_M=BLOCK_M_1,
        BLOCK_D=BLOCK_D_1,
    )
    
    BLOCK_M_2 = 128
    BLOCK_D_2 = BLOCK_D_1
    grid_2 = (triton.cdiv(M, BLOCK_M_2), triton.cdiv(D, BLOCK_D_2))
    
    _apply_quant_kernel[grid_2](
        v_2d, v_quant, scale, dummy_mean,
        M, D,
        v_2d.stride(0), v_2d.stride(1),
        v_quant.stride(0), v_quant.stride(1),
        SMOOTH_V=smooth_v,
        BLOCK_M=BLOCK_M_2,
        BLOCK_D=BLOCK_D_2,
    )
    
    v_quant = v_quant.view(v.shape)
    
    if smooth_v:
        return v_quant, scale, mean
    else:
        return v_quant, scale, None


# ==========================================
# 3. 验证与性能测试
# ==========================================
if __name__ == "__main__":
    # 配置张量
    B, H, N, D = 4, 32, 2048, 128
    device = "cuda"
    print(f"Testing tensor shape: {(B, H, N, D)}, Layout: BHND")
    
    # 模拟服从正态分布的 Attention Value 矩阵
    v = torch.randn((B, H, N, D), device=device, dtype=torch.float16) * 2.5 + 0.5
    
    smooth_mode = False
    print(f"--- Testing smooth_v={smooth_mode} ---")
    
    # 1. 精度对比
    v_q_ref, s_ref, m_ref = per_channel_int8_pytorch(v, smooth_v=smooth_mode)
    v_q_tri, s_tri, m_tri = per_channel_int8_triton(v, smooth_v=smooth_mode)
    # shape print
    print(f"v_q_ref shape: {v_q_ref.shape}")
    print(f"v_q_tri shape: {v_q_tri.shape}")
    print(f"s_ref shape: {s_ref.shape}")
    print(f"s_tri shape: {s_tri.shape}")
    if smooth_mode:
        print(f"m_ref shape: {m_ref.shape}")
        print(f"m_tri shape: {m_tri.shape}")
    
    # 使用 float32 计算差异防止 int8 溢出报错
    diff_q = torch.max(torch.abs(v_q_ref.float() - v_q_tri.float())).item()
    diff_s = torch.max(torch.abs(s_ref - s_tri)).item()
    
    print(f"[Accuracy] Max Quantization Diff (INT8): {diff_q}") 
    print(f"[Accuracy] Max Scale Diff (FP32): {diff_s:.8f}")
    
    if smooth_mode:
        diff_m = torch.max(torch.abs(m_ref - m_tri)).item()
        print(f"[Accuracy] Max Mean Diff (FP32): {diff_m:.8f}")

    if diff_q >= 1e-6:
        print("Quantization results mismatch!")
        print(torch.abs(v_q_ref - v_q_tri).sum())
        print(torch.abs(s_ref - s_tri).sum())
        if smooth_mode:
            print(torch.abs(m_ref - m_tri).sum())
    
    # import ipdb; ipdb.set_trace()
    if smooth_mode:
        dequant_error = (v_q_tri * s_tri + m_tri - v).abs().max()
        dequant_error_ref = (v_q_ref * s_ref + m_ref - v).abs().max()
        print(f"[Accuracy] Max Dequantization Diff (triton): {dequant_error:.8f}")
        print(f"[Accuracy] Max Dequantization Diff (ref): {dequant_error_ref:.8f}")
    else:
        dequant_error = (v_q_tri * s_tri - v).abs().max()
        dequant_error_ref = (v_q_ref * s_ref - v).abs().max()
        print(f"[Accuracy] Max Dequantization Diff (triton): {dequant_error:.8f}")
        print(f"[Accuracy] Max Dequantization Diff (ref): {dequant_error_ref:.8f}")
    
    # # 2. 预热与性能测试
    # warmup = 10
    # iters = 100
    
    # for _ in range(warmup):
    #     per_channel_int8_pytorch(v, smooth_v=smooth_mode)
    # torch.cuda.synchronize()
    
    # start = time.time()
    # for _ in range(iters):
    #     per_channel_int8_pytorch(v, smooth_v=smooth_mode)
    # torch.cuda.synchronize()
    # torch_time = (time.time() - start) / iters * 1000
    
    # for _ in range(warmup):
    #     per_channel_int8_triton(v, smooth_v=smooth_mode)
    # torch.cuda.synchronize()
    
    # start = time.time()
    # for _ in range(iters):
    #     per_channel_int8_triton(v, smooth_v=smooth_mode)
    # torch.cuda.synchronize()
    # triton_time = (time.time() - start) / iters * 1000
    
    # print(f"\n[Performance]")
    # print(f"PyTorch Time: {torch_time:.3f} ms")
    # print(f"Triton Time:  {triton_time:.3f} ms")
    # print(f"Speedup:      {torch_time / triton_time:.2f}x")