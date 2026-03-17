import torch
import triton
import triton.language as tl

@triton.jit
def _per_channel_int8_kernel(
    V_ptr, Out_ptr, Scale_ptr, Mean_ptr,
    stride_v_b, stride_v_0, stride_v_1, stride_v_d,
    stride_o_b, stride_o_0, stride_o_1, stride_o_d,
    stride_s_b, stride_s_0, stride_s_1,
    B, dim0, dim1, D,
    SMOOTH_V: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    # 1D grid，覆盖 Batch, dim0 (Head/Seq) 和 dim1 (Seq/Head)
    pid = tl.program_id(0)
    
    # 根据 pid 反推 B, dim0 和 dim1 的索引
    # 假设展开顺序为 B -> dim0 -> dim1
    idx1 = pid % dim1
    pid_b_dim0 = pid // dim1
    idx0 = pid_b_dim0 % dim0
    idx_b = pid_b_dim0 // dim0
    
    # 计算当前 batch/head/seq 对应的起始内存偏移量
    v_offset = idx_b * stride_v_b + idx0 * stride_v_0 + idx1 * stride_v_1
    o_offset = idx_b * stride_o_b + idx0 * stride_o_0 + idx1 * stride_o_1
    s_offset = idx_b * stride_s_b + idx0 * stride_s_0 + idx1 * stride_s_1
    
    # 生成 D 维度的索引和掩码 (防止 head_dim 不是 2 的幂次)
    d_offsets = tl.arange(0, BLOCK_D)
    mask = d_offsets < D
    
    # 加载向量
    v_ptrs = V_ptr + v_offset + d_offsets * stride_v_d
    v = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # 1. Smooth V (去均值)
    if SMOOTH_V:
        # 计算该通道的均值
        mean = tl.sum(v, axis=0) / D
        v = tl.where(mask, v - mean, 0.0)
        # 存储均值
        tl.store(Mean_ptr + s_offset, mean)
        
    # 2. 计算 Scale
    abs_v = tl.abs(v)
    max_val = tl.max(abs_v, axis=0)
    # 防止除以 0 的情况发生
    scale = tl.maximum(max_val / 127.0, 1e-12)
    tl.store(Scale_ptr + s_offset, scale)
    
    # 3. 量化并限制在 [-128, 127]
    v_quant = v / scale
    # 四舍五入 (Round)
    v_quant = tl.where(v_quant >= 0, v_quant + 0.5, v_quant - 0.5)
    v_quant = tl.maximum(-128.0, tl.minimum(127.0, v_quant))
    
    # 4. 转换为 int8 并写回显存
    v_quant_int8 = v_quant.to(tl.int8)
    o_ptrs = Out_ptr + o_offset + d_offsets * stride_o_d
    tl.store(o_ptrs, v_quant_int8, mask=mask)


def per_channel_int8_triton(
    v: torch.Tensor, 
    tensor_layout: str = "BHND", 
    smooth_v: bool = True
):
    """
    针对 4D 张量的最后一个维度(head_dim)进行 per-channel INT8 量化。
    支持格式如 'BHND' (Batch, Head, Seq, Dim) 或 'BNHD' (Batch, Seq, Head, Dim)。
    """
    assert v.dim() == 4, "输入张量必须是 4D 的 (包含 batch 维度)"
    assert v.is_cuda, "输入张量必须在 GPU 上"
    assert tensor_layout in ["BHND", "BNHD"], "布局参数应为包含 Batch 维度的 'BHND' 或 'BNHD'"
    
    # 提取 4D 形状与步长
    B, dim0, dim1, D = v.shape
    stride_v_b, stride_v_0, stride_v_1, stride_v_d = v.stride()
    
    # 分配输出内存
    v_quant = torch.empty_like(v, dtype=torch.int8)
    stride_o_b, stride_o_0, stride_o_1, stride_o_d = v_quant.stride()
    
    # 分配 Scale 和 Mean 内存，保留原来的 4D 形状但最后一个维度为 1
    stats_shape = (B, dim0, dim1, 1)
    scale = torch.empty(stats_shape, device=v.device, dtype=torch.float32)
    stride_s_b, stride_s_0, stride_s_1, _ = scale.stride()
    
    mean = None
    if smooth_v:
        mean = torch.empty(stats_shape, device=v.device, dtype=torch.float32)

    # 寻找大于等于 D 的下一个 2 的幂次方，用于分配 Shared Memory 大小
    BLOCK_D = triton.next_power_of_2(D)
    
    # Grid 设为前三个维度的乘积 (B * dim0 * dim1)
    # 不使用 3D grid 是为了避免 seq_len 超过 65535 时的 CUDA block 限制
    grid = lambda meta: (B * dim0 * dim1, )
    
    # 启动 Kernel
    _per_channel_int8_kernel[grid](
        v, v_quant, scale, mean,
        stride_v_b, stride_v_0, stride_v_1, stride_v_d,
        stride_o_b, stride_o_0, stride_o_1, stride_o_d,
        stride_s_b, stride_s_0, stride_s_1,
        B, dim0, dim1, D,
        SMOOTH_V=smooth_v,
        BLOCK_D=BLOCK_D
    )
    
    if smooth_v:
        return v_quant, scale, mean
    else:
        return v_quant, scale, None