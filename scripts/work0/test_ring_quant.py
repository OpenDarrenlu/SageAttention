from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse

def ring_quant_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_KV: int,
    tensor_layout: str = "NHD",
    causal=False,
    return_attn_probs=False,
):
    out = None
    lse = None
    next_k, next_v = None, None
    assert not causal, "only support non-causal attention"
    assert tensor_layout == "NHD", "only support NHD layout"
    batch_size, seqlen, num_heads, head_dim = k.shape
    # assert seqlen % block_KV == 0, "seqlen must be divisible by block_KV(TODO: fix)"
    num_blocks = (seqlen + block_KV - 1) // block_KV

    for step in range(num_blocks):
        # import ipdb; ipdb.set_trace()
        if  step & 0x1:
        # if  step % 3 == 0:
        # if  step % 2 == 0:
            # use sage attention
            from sageattention import sageattn
            block_out, block_lse = sageattn(
                q,
                k[:, step * block_KV : min((step + 1) * block_KV, seqlen), :, :],
                v[:, step * block_KV : min((step + 1) * block_KV, seqlen), :, :],
                tensor_layout,
                causal,
                return_lse=True,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            # use flash attn
            from flash_attn import flash_attn_func
            block_out, block_lse, _ = flash_attn_func(
                q,
                k[:, step * block_KV : min((step + 1) * block_KV, seqlen), :, :],
                v[:, step * block_KV : min((step + 1) * block_KV, seqlen), :, :],
                causal=causal,
                return_attn_probs=return_attn_probs,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

    # out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse if return_attn_probs else out

# test ring quant attn with flash attn
def test_ring_quant_attn_with_flash_attn():
    # batch_size, num_heads, seqlen, head_dim = 2, 4, 1024, 64
    # q = torch.randn(batch_size, seqlen, num_heads, head_dim, device="cuda").to(torch.float16)
    # k = torch.randn(batch_size, seqlen, num_heads, head_dim, device="cuda").to(torch.float16)
    # v = torch.randn(batch_size, seqlen, num_heads, head_dim, device="cuda").to(torch.float16)
    # 从 .pt 文件加载 q, k, v
    id = 0
    qkv = torch.load(f'../qkv_tensors_30layers/qkv_tensors_{id}.pt')
    # import ipdb; ipdb.set_trace()
    q,k,v = qkv["query"], qkv["key"], qkv["value"]
    batch_size, num_heads, seqlen, head_dim = q.shape
    print(f"q.shape: {q.shape}")
    q = q.to(device='cuda')
    k = k.to(device='cuda')
    v = v.to(device='cuda')
    # 转换为NHD布局
    q_nhd = q.permute(0, 2, 1, 3)
    k_nhd = k.permute(0, 2, 1, 3)
    v_nhd = v.permute(0, 2, 1, 3)
    block_KV = 8
    out, lse = ring_quant_attn(
        q_nhd, k_nhd, v_nhd, block_KV, tensor_layout="NHD", causal=False, return_attn_probs=True
    )
    assert out.shape == (batch_size, seqlen, num_heads, head_dim)
    assert lse.shape == (batch_size, num_heads, seqlen)
    # test accuracy
    from flash_attn import flash_attn_func
    flash_out, _, _ = flash_attn_func(
        q_nhd, k_nhd, v_nhd, causal=False, return_attn_probs=True
    )
    # cal cosine similarity
    cos_sim = F.cosine_similarity(out.to(torch.float32), flash_out.to(torch.float32), dim=-1)
    # cal max_error and relative error
    max_error = (out.to(torch.float32) - flash_out.to(torch.float32)).abs().max()
    rel_error = max_error / (flash_out.to(torch.float32).abs().max() + 1e-8)
    print(f"cos_sim: {cos_sim.mean().item():.4f}, max_error: {max_error.item():.4f}, rel_error: {rel_error.item():.4f}")
    # assert torch.allclose(out.to(torch.float32), flash_out.to(torch.float32), atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    test_ring_quant_attn_with_flash_attn()
'''
# random input:
sage and flash:
    cos_sim: 1.0000, max_error: 0.0055, rel_error: 0.0124
all flash:
    cos_sim: 1.0000, max_error: 0.0002, rel_error: 0.0005
all sage:
    cos_sim: 0.9999, max_error: 0.0052, rel_error: 0.0151

# real model input(&0x1, 64):
sage and flash:
    cos_sim: 1.0000, max_error: 0.1264, rel_error: 0.0300
all flash:
    cos_sim: 1.0000, max_error: 0.0184, rel_error: 0.0044
all sage:
    cos_sim: 0.9999, max_error: 0.1549, rel_error: 0.0367

# real model input(&0x1, 8):
sage and flash:
    cos_sim: 1.0000, max_error: 0.1415, rel_error: 0.0336

# real model input(&0x1, 2):
sage and flash:
    cos_sim: 1.0000, max_error: 0.1061, rel_error: 0.0251

# real model input(&0x1, 1):
sage and flash:
    cos_sim: 1.0000, max_error: 0.0949, rel_error: 0.0225

# real model input(%3==0, 1):
sage and flash:
    cos_sim: 1.0000, max_error: 0.0854, rel_error: 0.0203

# real model input(%4==0, 1):
sage and flash:
    cos_sim: 1.0000, max_error: 0.0859, rel_error: 0.0204

'''