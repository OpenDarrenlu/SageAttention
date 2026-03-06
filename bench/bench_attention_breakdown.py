import argparse
import math
import os
import sys
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from bench.utils import bench
except ImportError:
    from utils import bench

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sageattention import sageattn_qk_int4_pv_fp8_cuda
from sageattention import sm89_compile
from sageattention.core import get_int4_kernel_config, per_channel_fp8
from sageattention.triton.quant_per_thread import per_thread_int4


def parse_seqlens(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs).to(dtype).view(1, 1, seq_len, head_dim // 2)
    sin = torch.sin(freqs).to(dtype).view(1, 1, seq_len, head_dim // 2)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def format_table(title: str, rows: Iterable[Tuple[str, float]], total: Optional[float] = None) -> str:
    rows = list(rows)
    width = max(len(name) for name, _ in rows)
    header = [title, f"{'Operator'.ljust(width)} | Time (ms) | Share"]
    header.append(f"{'-' * width}-|----------:|------:")
    for name, value in rows:
        share = f"{(value / total * 100):6.2f}%" if total and total > 0 else "   n/a"
        header.append(f"{name.ljust(width)} | {value:9.3f} | {share}")
    if total is not None:
        header.append(f"{'Total'.ljust(width)} | {total:9.3f} | 100.00%")
    return "\n".join(header)


def make_weights(hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    scale = 0.02
    return {
        "wq": torch.randn(num_heads * head_dim, hidden_size, device=device, dtype=dtype) * scale,
        "wk": torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=dtype) * scale,
        "wv": torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=dtype) * scale,
        "wo": torch.randn(hidden_size, num_heads * head_dim, device=device, dtype=dtype) * scale,
    }


def reshape_q(x: torch.Tensor, batch_size: int, seqlen: int, num_heads: int, head_dim: int) -> torch.Tensor:
    return x.view(batch_size, seqlen, num_heads, head_dim).transpose(1, 2).contiguous()


def reshape_kv(x: torch.Tensor, batch_size: int, seqlen: int, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    return x.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2).contiguous()


def flatten_hnd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), x.size(1) * x.size(3))


def expand_kv_to_q_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    if num_heads % x.size(1) != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={x.size(1)}")
    return torch.repeat_interleave(x, num_heads // x.size(1), dim=1)


def maybe_causal(scores: torch.Tensor) -> torch.Tensor:
    seq_q, seq_k = scores.size(-2), scores.size(-1)
    mask = torch.triu(torch.ones((seq_q, seq_k), device=scores.device, dtype=torch.bool), diagonal=1)
    return scores.masked_fill(mask.view(1, 1, seq_q, seq_k), float("-inf"))


@torch.no_grad()
def benchmark_breakdown(
    batch_size: int,
    seqlen: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    num_warmups: int,
    num_tests: int,
    is_causal: bool,
    int4_kernel_config: Optional[int],
) -> Tuple[OrderedDict, OrderedDict, float, float]:
    hidden_size = num_heads * head_dim
    device = torch.device("cuda")
    weights = make_weights(hidden_size, num_heads, num_kv_heads, head_dim, device=device, dtype=dtype)
    hidden_states = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=dtype)
    cos, sin = build_rope_cache(seqlen, head_dim, device=device, dtype=dtype)
    sm_scale = head_dim ** -0.5

    def q_linear():
        return F.linear(hidden_states, weights["wq"])

    def k_linear():
        return F.linear(hidden_states, weights["wk"])

    def v_linear():
        return F.linear(hidden_states, weights["wv"])

    q_proj = reshape_q(q_linear(), batch_size, seqlen, num_heads, head_dim)
    k_proj = reshape_kv(k_linear(), batch_size, seqlen, num_kv_heads, head_dim)
    v_proj = reshape_kv(v_linear(), batch_size, seqlen, num_kv_heads, head_dim)

    def rope():
        return apply_rope(q_proj, cos, sin), apply_rope(k_proj, cos, sin)

    q_rope, k_rope = rope()
    k_rope_ref = expand_kv_to_q_heads(k_rope, num_heads)
    v_proj_ref = expand_kv_to_q_heads(v_proj, num_heads)

    def k_smooth():
        return k_rope.mean(dim=2, keepdim=True)

    km = k_smooth()

    kernel_config, blkq, warpq, blkk, warpk = get_int4_kernel_config(
        head_dim=head_dim,
        qo_len=seqlen,
        kv_len=seqlen,
        kernel_config=int4_kernel_config,
    )

    def qk_int4_quant():
        return per_thread_int4(
            q_rope,
            k_rope,
            km,
            tensor_layout="HND",
            BLKQ=blkq,
            WARPQ=warpq,
            BLKK=blkk,
            WARPK=warpk,
        )

    def v_fp8_quant():
        return per_channel_fp8(v_proj, tensor_layout="HND", scale_max=2.25, smooth_v=False)

    q_int4, q_scale, k_int4, k_scale = qk_int4_quant()
    v_fp8, v_scale, _ = v_fp8_quant()
    output = torch.empty_like(q_rope)

    def sage_core():
        return sm89_compile.qk_int4_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int4,
            k_int4,
            v_fp8,
            output,
            q_scale,
            k_scale,
            v_scale,
            1,
            1 if is_causal else 0,
            3,
            kernel_config,
            sm_scale,
            0,
        )

    attn_out = sageattn_qk_int4_pv_fp8_cuda(
        q_rope,
        k_rope,
        v_proj,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran="per_thread",
        pv_accum_dtype="fp32+fp16",
        smooth_k=True,
        int4_kernel_config=kernel_config,
    )
    attn_out_flat = flatten_hnd(attn_out)

    def o_linear():
        return F.linear(attn_out_flat, weights["wo"])

    def full_sage_block():
        q = reshape_q(F.linear(hidden_states, weights["wq"]), batch_size, seqlen, num_heads, head_dim)
        k = reshape_kv(F.linear(hidden_states, weights["wk"]), batch_size, seqlen, num_kv_heads, head_dim)
        v = reshape_kv(F.linear(hidden_states, weights["wv"]), batch_size, seqlen, num_kv_heads, head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = sageattn_qk_int4_pv_fp8_cuda(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=is_causal,
            qk_quant_gran="per_thread",
            pv_accum_dtype="fp32+fp16",
            smooth_k=True,
            int4_kernel_config=kernel_config,
        )
        return F.linear(flatten_hnd(out), weights["wo"])

    def qk_matmul():
        scores = torch.matmul(q_rope, k_rope_ref.transpose(-1, -2)).float() * sm_scale
        return maybe_causal(scores) if is_causal else scores

    scores = qk_matmul()

    def softmax():
        return torch.softmax(scores, dim=-1)

    probs = softmax().to(dtype)

    def pv_matmul():
        return torch.matmul(probs, v_proj_ref)

    def full_reference_block():
        q = reshape_q(F.linear(hidden_states, weights["wq"]), batch_size, seqlen, num_heads, head_dim)
        k = reshape_kv(F.linear(hidden_states, weights["wk"]), batch_size, seqlen, num_kv_heads, head_dim)
        v = reshape_kv(F.linear(hidden_states, weights["wv"]), batch_size, seqlen, num_kv_heads, head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k_ref = expand_kv_to_q_heads(k, num_heads)
        v_ref = expand_kv_to_q_heads(v, num_heads)
        reference_scores = torch.matmul(q, k_ref.transpose(-1, -2)).float() * sm_scale
        if is_causal:
            reference_scores = maybe_causal(reference_scores)
        reference_probs = torch.softmax(reference_scores, dim=-1).to(dtype)
        out = torch.matmul(reference_probs, v_ref)
        return F.linear(flatten_hnd(out), weights["wo"])

    sage_rows = OrderedDict([
        ("q_linear", bench(q_linear, num_warmups=num_warmups, num_tests=num_tests)),
        ("k_linear", bench(k_linear, num_warmups=num_warmups, num_tests=num_tests)),
        ("v_linear", bench(v_linear, num_warmups=num_warmups, num_tests=num_tests)),
        ("rope_qk", bench(rope, num_warmups=num_warmups, num_tests=num_tests)),
        ("k_smooth_mean", bench(k_smooth, num_warmups=num_warmups, num_tests=num_tests)),
        ("qk_int4_quant", bench(qk_int4_quant, num_warmups=num_warmups, num_tests=num_tests)),
        ("v_fp8_quant", bench(v_fp8_quant, num_warmups=num_warmups, num_tests=num_tests)),
        ("sage_core_fused", bench(sage_core, num_warmups=num_warmups, num_tests=num_tests)),
        ("o_linear", bench(o_linear, num_warmups=num_warmups, num_tests=num_tests)),
    ])
    reference_rows = OrderedDict([
        ("q_linear", sage_rows["q_linear"]),
        ("k_linear", sage_rows["k_linear"]),
        ("v_linear", sage_rows["v_linear"]),
        ("rope_qk", sage_rows["rope_qk"]),
        ("qk_matmul", bench(qk_matmul, num_warmups=num_warmups, num_tests=num_tests)),
        ("softmax", bench(softmax, num_warmups=num_warmups, num_tests=num_tests)),
        ("pv_matmul", bench(pv_matmul, num_warmups=num_warmups, num_tests=num_tests)),
        ("o_linear", sage_rows["o_linear"]),
    ])

    full_sage_ms = bench(full_sage_block, num_warmups=num_warmups, num_tests=num_tests)
    full_reference_ms = bench(full_reference_block, num_warmups=num_warmups, num_tests=num_tests)
    return sage_rows, reference_rows, full_sage_ms, full_reference_ms


def main():
    parser = argparse.ArgumentParser(description="Profile attention block breakdown for SageAttention on sm89")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    parser.add_argument("--seqlens", type=str, default="2048,4096")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--num_warmups", type=int, default=10)
    parser.add_argument("--num_tests", type=int, default=30)
    parser.add_argument("--int4-kernel-config", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 9):
        raise SystemExit(f"This benchmark targets sm89 (RTX 4090 class), got sm{capability[0]}{capability[1]}")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    seqlens = parse_seqlens(args.seqlens)

    print("SageAttention operator breakdown on sm89")
    print(
        f"batch={args.batch_size}, heads={args.num_heads}, kv_heads={args.num_kv_heads}, "
        f"head_dim={args.head_dim}, dtype={args.dtype}, causal={args.is_causal}, seqlens={seqlens}"
    )
    print("Note: SageAttention's softmax is fused inside `sage_core_fused`; the standalone `softmax` row below comes from an unfused reference path.")

    for seqlen in seqlens:
        sage_rows, reference_rows, full_sage_ms, full_reference_ms = benchmark_breakdown(
            batch_size=args.batch_size,
            seqlen=seqlen,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            num_warmups=args.num_warmups,
            num_tests=args.num_tests,
            is_causal=args.is_causal,
            int4_kernel_config=args.int4_kernel_config,
        )

        sage_total = sum(sage_rows.values())
        reference_total = sum(reference_rows.values())

        print()
        print(f"=== seqlen={seqlen} ===")
        print(format_table("SageAttention INT4 pipeline", sage_rows.items(), total=sage_total))
        print(f"Measured full SageAttention block: {full_sage_ms:.3f} ms")
        print()
        print(format_table("Reference unfused attention pipeline", reference_rows.items(), total=reference_total))
        print(f"Measured full unfused reference block: {full_reference_ms:.3f} ms")
        print()
        print(
            f"Speedup (full Sage block vs unfused reference): {full_reference_ms / full_sage_ms:.3f}x | "
            f"INT4 quantized preprocessing share: {(sage_rows['k_smooth_mean'] + sage_rows['qk_int4_quant'] + sage_rows['v_fp8_quant']) / sage_total * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
