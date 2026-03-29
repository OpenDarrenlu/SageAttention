"""
Wan（diffusers WanTransformer3DModel）注意力替换为 sageattention/core_Pcodebook.py 中的实现，并对照测试。

参考实现使用包内标准 ``sageattn``（与 example/modify_model/modify_wan1.py 中对比 sageattn / sageattn_pint 的思路一致）。

用法（需在仓库根目录或设置 PYTHONPATH 含仓库根）:
  python scripts/work2/modify_wan2.py

依赖: torch, diffusers, sageattention（本仓库）, torchmm, CUDA；可选 P_cookbook（同目录）。
"""

from __future__ import annotations

import os
import sys
from functools import partial
from typing import Callable, Optional

# 仓库根目录 → import sageattention
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models import WanTransformer3DModel

from sageattention.core_Pcodebook import sageattn_pcodebook_torch


class WanAttnProcessor2_0:
    """与 example/modify_model/modify_wan1.py 一致：将 self-attention 交给外部 attn_func。"""

    def __init__(self, attn_func: Callable):
        self.attn_func = attn_func
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

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


def _sageattn_pcodebook_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    p_cookbook_manager=None,
    p_codebook_layer_idx: int = 0,
    **kwargs,
) -> torch.Tensor:
    """
    适配 diffusers 的 scaled_dot_product_attention 风格调用；
    core_Pcodebook 当前未使用 is_causal / dropout_p（推理常为 0）。
    """
    if is_causal:
        raise NotImplementedError(
            "core_Pcodebook.sageattn_pcodebook_torch 路径未接 is_causal，请勿对因果注意力使用本入口。"
        )
    _ = dropout_p
    kw = dict(tensor_layout="HND", attn_mask=attn_mask, **kwargs)
    if p_cookbook_manager is not None:
        kw["p_cookbook_manager"] = p_cookbook_manager
        kw["p_codebook_layer_idx"] = p_codebook_layer_idx
    return sageattn_pcodebook_torch(q, k, v, **kw)


def set_sage_attn_wan_pcodebook(
    model: WanTransformer3DModel,
    p_cookbook_manager=None,
) -> None:
    """
    将每个 block.attn1.processor 换为 WanAttnProcessor2_0，内部调用 sageattention.core_Pcodebook.sageattn_pcodebook_torch。

    p_cookbook_manager:
      None — 仅 INT8 QK + 高精度 PV，不启用 P cookbook。
      PCookbookManager — 需已对该模型层数 seed_placeholder_cookbooks(range(len(blocks))) 或 load；
      每层使用对应 layer_idx，forward 时 observe，推理结束后可 finalize_all 更新码本。
    """
    n_blocks = len(model.blocks)
    for idx, block in enumerate(model.blocks):
        fn = partial(
            _sageattn_pcodebook_sdpa,
            p_cookbook_manager=p_cookbook_manager,
            p_codebook_layer_idx=idx,
        )
        block.attn1.processor = WanAttnProcessor2_0(fn)


def _run_compare_test() -> None:
    if not torch.cuda.is_available():
        print("跳过测试：需要 CUDA（sageattention.core_Pcodebook 要求 GPU）。")
        return

    device = torch.device("cuda")
    from sageattention import sageattn

    IN_CHANNELS = 16
    OUT_CHANNELS = 16
    NUM_LAYERS = 2
    NUM_FRAMES = 4
    WIDTH = 64
    HEIGHT = 64

    def build_model():
        return WanTransformer3DModel(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            attention_head_dim=128,
            num_attention_heads=12,
            num_layers=NUM_LAYERS,
        ).to(torch.bfloat16).to(device)

    x = torch.randn(1, IN_CHANNELS, NUM_FRAMES, WIDTH, HEIGHT, device=device, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 257 + IN_CHANNELS, 4096, device=device, dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (1,), device=device, dtype=torch.bfloat16)

    # 1) 参考：包内 sageattn（INT8 QK + FP16 PV，见 sageattention.core.sageattn）
    model_ref = build_model()

    def sdpa_sageattn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        _ = dropout_p
        # sageattn 入口未转发 attn_mask 至子内核时，与 Wan 常见全注意力（mask=None）一致
        return sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal)

    for block in model_ref.blocks:
        block.attn1.processor = WanAttnProcessor2_0(sdpa_sageattn)

    with torch.no_grad():
        out_ref = model_ref(hidden_states=x, encoder_hidden_states=encoder_hidden_states, timestep=timestep)[0]
    print(f"参考 sageattn 输出 shape: {out_ref.shape}")

    # 2) core_Pcodebook（无 P cookbook）
    model_pc = build_model()
    set_sage_attn_wan_pcodebook(model_pc, p_cookbook_manager=None)
    with torch.no_grad():
        out_pc = model_pc(hidden_states=x, encoder_hidden_states=encoder_hidden_states, timestep=timestep)[0]
    print(f"core_Pcodebook 输出 shape: {out_pc.shape}")

    diff = (out_ref - out_pc).abs()
    print(
        f"与 sageattn 最大绝对误差: {diff.max().item():.6f} "
        f"(ref max {out_ref.abs().max().item():.6f}, core_Pcodebook max {out_pc.abs().max().item():.6f})"
    )

    # 3) 带 PCookbookManager：两轮 forward → finalize → 再 forward（码本已更新）
    _work2 = os.path.join(_REPO_ROOT, "scripts", "work2")
    if _work2 not in sys.path:
        sys.path.insert(0, _work2)
    import P_cookbook as pc  # noqa: WPS433

    mgr = pc.PCookbookManager(bits=4, importance_exp=1.0, max_samples_per_layer=500_000)
    mgr.seed_placeholder_cookbooks(list(range(NUM_LAYERS)))

    model_cb = build_model()
    set_sage_attn_wan_pcodebook(model_cb, p_cookbook_manager=mgr)
    with torch.no_grad():
        _ = model_cb(hidden_states=x, encoder_hidden_states=encoder_hidden_states, timestep=timestep)[0]
        _ = model_cb(hidden_states=x, encoder_hidden_states=encoder_hidden_states, timestep=timestep)[0]
    mgr.finalize_all()
    mgr.save("mgr.pt")
    with torch.no_grad():
        out_cb = model_cb(hidden_states=x, encoder_hidden_states=encoder_hidden_states, timestep=timestep)[0]
    print(f"P cookbook finalize 后输出 shape: {out_cb.shape}")
    diff_cb = (out_ref - out_cb).abs().max().item()
    print(f"与 sageattn 最大绝对误差（cookbook 路径）: {diff_cb:.6f}")

    assert out_ref.shape == x.shape == out_pc.shape == out_cb.shape
    print("modify_wan2 测试完成。")


if __name__ == "__main__":
    _run_compare_test()
