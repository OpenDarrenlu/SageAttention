import argparse
import os
import sys
from typing import List

import torch

try:
    from bench.utils import bench
except ImportError:
    from utils import bench

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sageattention.lut_softmax import lut_attention, lut_softmax


def parse_seqlens(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Benchmark LUT softmax and LUT attention reference against standard softmax attention")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    parser.add_argument("--seqlens", type=str, default="1024,2048,4096")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--clip_value", type=float, default=4.5)
    parser.add_argument("--lut_entries", type=int, default=32)
    parser.add_argument("--q_chunk_size", type=int, default=512)
    parser.add_argument("--num_warmups", type=int, default=10)
    parser.add_argument("--num_tests", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    seqlens = parse_seqlens(args.seqlens)
    torch.manual_seed(args.seed)

    print("LUT softmax benchmark")
    print(
        f"batch={args.batch_size}, heads={args.num_heads}, head_dim={args.head_dim}, "
        f"dtype={args.dtype}, causal={args.is_causal}, clip_value={args.clip_value}, lut_entries={args.lut_entries}"
    )

    for seqlen in seqlens:
        q = torch.randn(args.batch_size, args.num_heads, seqlen, args.head_dim, device=device, dtype=dtype)
        k = torch.randn(args.batch_size, args.num_heads, seqlen, args.head_dim, device=device, dtype=dtype)
        v = torch.randn(args.batch_size, args.num_heads, seqlen, args.head_dim, device=device, dtype=dtype)
        sm_scale = args.head_dim ** -0.5

        scores = torch.matmul(q, k.transpose(-1, -2)).to(torch.float32) * sm_scale
        if args.is_causal:
            mask = torch.triu(torch.ones((seqlen, seqlen), device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.view(1, 1, seqlen, seqlen), float("-inf"))

        def ref_softmax():
            return torch.softmax(scores, dim=-1)

        def lut_softmax_fn():
            return lut_softmax(scores, clip_value=args.clip_value, num_entries=args.lut_entries)

        def ref_attention():
            probs = torch.softmax(scores, dim=-1).to(dtype)
            return torch.matmul(probs, v)

        def lut_attention_fn():
            return lut_attention(
                q,
                k,
                v,
                sm_scale=sm_scale,
                is_causal=args.is_causal,
                clip_value=args.clip_value,
                num_entries=args.lut_entries,
                q_chunk_size=args.q_chunk_size,
            )

        estimate_scores_bytes = scores.numel() * 4
        if estimate_scores_bytes <= 1_200_000_000:
            ms_ref_softmax = bench(ref_softmax, num_warmups=args.num_warmups, num_tests=args.num_tests)
            ms_lut_softmax = bench(lut_softmax_fn, num_warmups=args.num_warmups, num_tests=args.num_tests)
            softmax_msg = f"softmax: torch={ms_ref_softmax:>7.3f} ms, lut={ms_lut_softmax:>7.3f} ms, speedup={ms_ref_softmax / ms_lut_softmax:>5.3f}x"
        else:
            ms_ref_softmax = None
            ms_lut_softmax = None
            softmax_msg = "softmax: skipped (score tensor too large for standalone benchmark)"
        ms_ref_attention = bench(ref_attention, num_warmups=args.num_warmups, num_tests=args.num_tests)
        ms_lut_attention = bench(lut_attention_fn, num_warmups=args.num_warmups, num_tests=args.num_tests)

        ref_probs = ref_softmax()
        approx_probs_u8 = lut_softmax(scores, clip_value=args.clip_value, num_entries=args.lut_entries, return_uint8=True)
        approx_probs = approx_probs_u8.to(torch.float32) / 255.0
        ref_out = ref_attention()
        approx_out = lut_attention_fn()

        probs_diff = (ref_probs - approx_probs).abs()
        out_diff = (ref_out - approx_out).abs()
        row_sums = approx_probs_u8.to(torch.int32).sum(dim=-1).to(torch.float32)

        print(
            f"seqlen={seqlen:>5d} | "
            f"{softmax_msg} | "
            f"attention: torch={ms_ref_attention:>7.3f} ms, lut={ms_lut_attention:>7.3f} ms, speedup={ms_ref_attention / ms_lut_attention:>5.3f}x"
        )
        print(
            f"             prob_mean_abs_diff={probs_diff.mean().item():.6f}, prob_max_abs_diff={probs_diff.max().item():.6f}, "
            f"out_mean_abs_diff={out_diff.mean().item():.6f}, out_max_abs_diff={out_diff.max().item():.6f}, "
            f"uint8_row_sum_mean={row_sums.mean().item():.2f}"
        )

        del q, k, v, scores, ref_probs, approx_probs_u8, approx_probs, ref_out, approx_out, probs_diff, out_diff, row_sums
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
