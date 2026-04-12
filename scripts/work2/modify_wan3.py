import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention
from diffusers.models import WanTransformer3DModel
from functools import partial

# 使用 P codebook 量化的 Attention
# 预训练的 centroids（从 results.txt 中获取）
p_codebook_centroids = torch.tensor([
    0.0000, 0.0609, 0.1123, 0.1715, 0.2381, 0.3104, 0.3859, 0.4637,
    0.5436, 0.6242, 0.7028, 0.7780, 0.8483, 0.9104, 0.9616, 0.9961
], dtype=torch.float32)

def quantize_p(p: torch.Tensor, centroids: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """量化 P 到最近的质心索引 - 使用分块处理避免 OOM"""
    c = centroids.to(device=p.device, dtype=p.dtype)
    
    # 如果张量较小，直接处理
    if p.numel() < chunk_size * 16:  # 简单阈值
        dist = (p.unsqueeze(-1) - c).abs()
        return dist.argmin(dim=-1)
    
    # 否则分块处理
    flat_p = p.flatten()
    indices_flat = torch.empty_like(flat_p, dtype=torch.long)
    
    for i in range(0, flat_p.numel(), chunk_size):
        chunk = flat_p[i:i+chunk_size]
        dist = (chunk.unsqueeze(-1) - c).abs()
        indices_flat[i:i+chunk_size] = dist.argmin(dim=-1)
    
    return indices_flat.reshape(p.shape)

def quantize_p_fast(p: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """快速量化方法：利用 centroids 已排序的特性，使用搜索代替距离计算"""
    c = centroids.to(device=p.device, dtype=p.dtype)
    
    # 因为 centroids 是升序排列的，我们可以使用搜索
    # 找到每个 p 应该插入的位置，然后比较相邻的两个
    indices = torch.searchsorted(c, p)
    
    # 处理边界情况
    indices = torch.clamp(indices, 1, len(c) - 1)
    
    # 比较当前位置和前一个位置，选择更近的
    left_dist = (p - c[indices - 1]).abs()
    right_dist = (p - c[indices]).abs()
    
    # 选择更近的那个
    mask = left_dist < right_dist
    indices[mask] = indices[mask] - 1
    
    return indices

def dequantize_p(indices: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """从索引反量化回 P"""
    return centroids.to(device=indices.device, dtype=torch.float32)[indices.long()]
            
def p_codebook_attn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """
    使用 P codebook 量化的 multi-head attention.
    Input shape: [batch, seq_len, num_heads, head_dim]
    """
    # Scaled dot-product attention
    attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [b, n, s, s]
    
    # scale by sqrt(d)
    attn_weights = attn_weights / (q.size(-1) ** 0.5)

    # Apply softmax -> outputs sum to 1, values in (0, 1)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # 使用 P codebook 量化 - 使用快速方法
    indices = quantize_p_fast(attn_weights, p_codebook_centroids)
    attn_weights_quant = dequantize_p(indices, p_codebook_centroids)

    # Compute output with quantized P
    output = torch.matmul(attn_weights_quant, v.to(attn_weights_quant.dtype))  # [b, n, s, d]
    
    return output
            
class WanAttnProcessor2_0:
    def __init__(self, attn_func):
        self.attn_func = attn_func
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = self.attn_func(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)
        
        hidden_states = self.attn_func(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
    
def set_sage_attn_wan(
        model: WanTransformer3DModel,
        attn_func,
):
    for idx, block in enumerate(model.blocks):
        processor = WanAttnProcessor2_0(attn_func)
        block.attn1.processor = processor

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # from sageattention import sageattn, sageattn_pint
    
    # test WanAttnProcessor2_0 with a dummy WanTransformer3DModel
    from diffusers import WanTransformer3DModel
    IN_CHANNELS = 4
    OUT_CHANNELS = 4
    NUM_LAYERS = 2
    NUM_FRAMES = 4  
    WIDTH = 64
    HEIGHT = 64
    model = WanTransformer3DModel(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        attention_head_dim=128,
        num_attention_heads=12,
        num_layers=NUM_LAYERS,
    ).to(torch.bfloat16).to(device)
    set_sage_attn_wan(model, F.scaled_dot_product_attention)
    # set_sage_attn_wan(model, sageattn_pint)
    x = torch.randn(1, IN_CHANNELS, NUM_FRAMES, WIDTH, HEIGHT).to(torch.bfloat16).to(device)
    encoder_hidden_states = torch.randn(1, 257 + IN_CHANNELS, 4096).to(torch.bfloat16).to(device)
    timestep = torch.randint(0, 1000, (1,)).to(torch.bfloat16).to(device)
    
    output = model(
        hidden_states=x,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep
    )[0]
    print(output.shape)  # should be [1, OUT_CHANNELS, NUM_FRAMES, WIDTH, HEIGHT]
    assert output.shape == x.shape
    # test sage attention and compare with original attention
    # set_sage_attn_wan(model, sageattn)
    set_sage_attn_wan(model, p_codebook_attn)
    out_sage = model(x, encoder_hidden_states=encoder_hidden_states, timestep=timestep)[0]
    
    # assert torch.allclose(output, out_sage, rtol=1e-2, atol=1e-2)
    # get the difference between output and out_sage
    diff = torch.abs(output - out_sage)
    print(f"Max difference between original(max: {output.max()}) and sage(max: {out_sage.max()}) attention: {diff.max()}")
    relative_error = diff.max() / torch.abs(output).max()
    print(f"Relative error: {relative_error}")
