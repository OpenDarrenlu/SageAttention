"""
Wan attention processor for the SageAttention LUT precision sweep experiment.

This module mirrors ``scripts/work3/modify_wan3.py`` but allows the
attention function to be parameterized by the P and V quantization precisions
used by ``sageattention.core_lut.sageattn_lut``.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional, Tuple
from diffusers.models.attention_processor import Attention
from diffusers.models import WanTransformer3DModel


class WanAttnProcessor2_0:
    """
    Diffusers attention processor for Wan that delegates to a custom
    attention callable.

    Compatible with diffusers >= 0.38, where ``rotary_emb`` is a tuple of
    ``(freqs_cos, freqs_sin)``.
    """

    def __init__(self, attn_func: Callable):
        self.attn_func = attn_func
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0+")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # Diffusers >= 0.38 uses 512 image-context tokens for Wan I2V.
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V image cross-attention branch
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = self.attn_func(
                query.transpose(1, 2), key_img.transpose(1, 2), value_img.transpose(1, 2),
                attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # sageattn_lut expects HND layout [B, heads, seq, head_dim]
        hidden_states = self.attn_func(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def make_sageattn_lut(p_quant_dtype: str = "fp16", v_quant_dtype: str = "int8", p_block_n: int = 64, v_block_size: int = 0) -> Callable:
    """
    Build an attention callable for ``WanAttnProcessor2_0`` that uses
    ``sageattn_lut`` with the requested P/V precisions.

    The signature matches ``F.scaled_dot_product_attention`` so it can be
    dropped into the existing processor.
    """
    from sageattention import sageattn_lut

    def attn_func(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        # sageattn_lut does not accept dropout_p / is_causal; they are ignored
        # because this experiment evaluates a deterministic diffusion model.
        return sageattn_lut(
            query, key, value,
            tensor_layout="HND",
            p_quant_dtype=p_quant_dtype,
            v_quant_dtype=v_quant_dtype,
            p_block_n=p_block_n,
            v_block_size=v_block_size,
        )

    return attn_func


def set_sage_attn_wan(
    model: WanTransformer3DModel,
    p_quant_dtype: str = "fp16",
    v_quant_dtype: str = "int8",
    p_block_n: int = 64,
    v_block_size: int = 0,
):
    """
    Patch every transformer block in ``model`` to use the SageAttention LUT
    processor with the requested P/V precisions.
    """
    attn_func = make_sageattn_lut(p_quant_dtype, v_quant_dtype, p_block_n, v_block_size)
    for block in model.blocks:
        block.attn1.processor = WanAttnProcessor2_0(attn_func)


def build_dummy_wan(
    in_channels: int = 16,
    out_channels: int = 16,
    num_layers: int = 2,
    attention_head_dim: int = 64,
    num_attention_heads: int = 12,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Construct a tiny WanTransformer3DModel for fast accuracy experiments.
    The architecture is real; only the layer count and spatial resolution are
    shrunk so the sweep finishes quickly.
    """
    model = WanTransformer3DModel(
        in_channels=in_channels,
        out_channels=out_channels,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        num_layers=num_layers,
    ).to(dtype).to(device)
    return model


def run_dummy_wan(
    model: WanTransformer3DModel,
    attn_func: Optional[Callable] = None,
    p_quant_dtype: str = "fp16",
    v_quant_dtype: str = "int8",
    p_block_n: int = 64,
    v_block_size: int = 0,
    num_frames: int = 4,
    width: int = 64,
    height: int = 64,
    in_channels: int = 16,
    seed: int = 42,
):
    """
    Run the dummy Wan model once with the specified attention callable.

    If ``attn_func`` is None, a SageAttention LUT callable is built from
    ``p_quant_dtype`` / ``v_quant_dtype`` / ``p_block_n`` / ``v_block_size``.
    """
    if attn_func is None:
        set_sage_attn_wan(model, p_quant_dtype, v_quant_dtype, p_block_n, v_block_size)
    else:
        for block in model.blocks:
            block.attn1.processor = WanAttnProcessor2_0(attn_func)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    torch.manual_seed(seed)
    x = torch.randn(1, in_channels, num_frames, width, height, dtype=dtype, device=device)
    encoder_hidden_states = torch.randn(1, 512 + in_channels, 4096, dtype=dtype, device=device)
    timestep = torch.randint(0, 1000, (1,), device=device).to(dtype)

    with torch.no_grad():
        output = model(
            hidden_states=x,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )[0]
    return output


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_dummy_wan(num_layers=2, attention_head_dim=64, device=device)

    # Baseline: original SDPA
    out_sdpa = run_dummy_wan(model, attn_func=F.scaled_dot_product_attention)
    print("SDPA output shape:", out_sdpa.shape)

    # Sage LUT baseline
    out_sage = run_dummy_wan(model, p_quant_dtype="fp16", v_quant_dtype="int8")
    diff = torch.abs(out_sdpa - out_sage)
    print(f"Max diff vs SDPA: {diff.max().item():.4f}")
    print(f"Mean diff vs SDPA: {diff.mean().item():.6f}")

    # One low-bit combo
    out_low = run_dummy_wan(model, p_quant_dtype="int4", v_quant_dtype="int4")
    diff = torch.abs(out_sdpa - out_low)
    print(f"Max diff P=int4 V=int4 vs SDPA: {diff.max().item():.4f}")
