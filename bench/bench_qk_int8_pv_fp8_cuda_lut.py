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

from sageattention import sageattn_qk_int8_pv_fp8_cuda, sageattn_qk_int8_pv_fp8_cuda_lut


def parse_seqlens(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def tops(batch: int, heads: int, head_dim: int, seqlen: int, ms: float, is_causal: bool) -> float:
    flops = 4.0 * batch * heads * head_dim * seqlen * seqlen
    if is_causal:
        flops *= 0.5
    return flops / (ms * 1e-3) / 1e12


def main():
    parser = argparse.ArgumentParser(description="Benchmark experimental LUT-softmax INT8 SageAttention against the original INT8 path on sm89")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    parser.add_argument("--seqlens", type=str, default="1024,2048,4096")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--num_warmups", type=int, default=20)
    parser.add_argument("--num_tests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 9):
        raise SystemExit(f"This benchmark targets sm89 (RTX 4090 class), got sm{capability[0]}{capability[1]}")

    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    seqlens = parse_seqlens(args.seqlens)

    print("SageAttention sm89 INT8 LUT-softmax benchmark")
    print(f"batch={args.batch_size}, heads={args.num_heads}, head_dim={args.head_dim}, dtype={args.dtype}, causal={args.is_causal}")
    print(f"warmups={args.num_warmups}, tests={args.num_tests}, seqlens={seqlens}")

    for seqlen in seqlens:
        q = torch.randn(args.batch_size, args.num_heads, seqlen, args.head_dim, dtype=dtype, device="cuda")
        k = torch.randn(args.batch_size, args.num_heads, seqlen, args.head_dim, dtype=dtype, device="cuda")
        v = torch.randn(args.batch_size, args.num_heads, seqlen, args.head_dim, dtype=dtype, device="cuda")

        def fn_ref():
            return sageattn_qk_int8_pv_fp8_cuda(
                q, k, v,
                tensor_layout="HND",
                is_causal=args.is_causal,
                qk_quant_gran="per_thread",
                pv_accum_dtype="fp32+fp16",
            )

        def fn_lut():
            return sageattn_qk_int8_pv_fp8_cuda_lut(
                q, k, v,
                tensor_layout="HND",
                is_causal=args.is_causal,
                qk_quant_gran="per_thread",
                pv_accum_dtype="fp32+fp16",
            )

        ms_ref = bench(fn_ref, num_warmups=args.num_warmups, num_tests=args.num_tests, high_precision=False)
        ms_lut = bench(fn_lut, num_warmups=args.num_warmups, num_tests=args.num_tests, high_precision=False)

        out_ref = fn_ref()
        out_lut = fn_lut()
        diff = (out_ref - out_lut).abs()

        tops_ref = tops(args.batch_size, args.num_heads, args.head_dim, seqlen, ms_ref, args.is_causal)
        tops_lut = tops(args.batch_size, args.num_heads, args.head_dim, seqlen, ms_lut, args.is_causal)
        speedup = ms_ref / ms_lut

        print(
            f"seqlen={seqlen:>5d} | "
            f"orig_int8={ms_ref:>7.3f} ms ({tops_ref:>6.2f} TOPS) | "
            f"lut_int8={ms_lut:>7.3f} ms ({tops_lut:>6.2f} TOPS) | "
            f"speedup={speedup:>5.3f}x | "
            f"mean_abs_diff={diff.mean().item():.5f}, max_abs_diff={diff.max().item():.5f}"
        )


if __name__ == "__main__":
    main()
