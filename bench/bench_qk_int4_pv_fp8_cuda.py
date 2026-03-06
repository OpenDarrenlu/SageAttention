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

from sageattention import sageattn_qk_int4_pv_fp8_cuda, sageattn_qk_int8_pv_fp8_cuda


def parse_seqlens(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int4_kernel_configs(raw: str) -> List[int]:
    if raw == "all":
        return [0, 1, 2]
    if raw == "auto":
        return []
    return [int(raw)]


def tops(batch: int, heads: int, head_dim: int, seqlen: int, ms: float, is_causal: bool) -> float:
    flops = 4.0 * batch * heads * head_dim * seqlen * seqlen
    if is_causal:
        flops *= 0.5
    return flops / (ms * 1e-3) / 1e12


def main():
    parser = argparse.ArgumentParser(description="Benchmark SageAttention sm89 int4 branch against int8 baseline")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    parser.add_argument("--seqlens", type=str, default="1024,2048,4096,8192")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--num_warmups", type=int, default=20)
    parser.add_argument("--num_tests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--int4-kernel-config", type=str, default="auto", help="auto, all, or an explicit int4 kernel config id")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 9):
        raise SystemExit(f"This benchmark targets sm89 (RTX 4090 class), got sm{capability[0]}{capability[1]}")

    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    seqlens = parse_seqlens(args.seqlens)
    int4_kernel_configs = parse_int4_kernel_configs(args.int4_kernel_config)

    print("SageAttention sm89 int4 benchmark")
    print(f"batch={args.batch_size}, heads={args.num_heads}, head_dim={args.head_dim}, dtype={args.dtype}, causal={args.is_causal}")
    print(f"warmups={args.num_warmups}, tests={args.num_tests}, seqlens={seqlens}, int4_kernel_config={args.int4_kernel_config}")

    for seqlen in seqlens:
        q = torch.randn(args.batch_size, args.num_heads, seqlen, args.head_dim, dtype=dtype, device="cuda")
        k = torch.randn(args.batch_size, args.num_heads, seqlen, args.head_dim, dtype=dtype, device="cuda")
        v = torch.randn(args.batch_size, args.num_heads, seqlen, args.head_dim, dtype=dtype, device="cuda")

        def fn_int8():
            return sageattn_qk_int8_pv_fp8_cuda(
                q, k, v,
                tensor_layout="HND",
                is_causal=args.is_causal,
                qk_quant_gran="per_thread",
                pv_accum_dtype="fp32+fp16",
            )

        ms_int8 = bench(fn_int8, num_warmups=args.num_warmups, num_tests=args.num_tests, high_precision=False)
        out_int8 = fn_int8()
        tops_int8 = tops(args.batch_size, args.num_heads, args.head_dim, seqlen, ms_int8, args.is_causal)
        print(f"seqlen={seqlen:>5d} | int8={ms_int8:>7.3f} ms ({tops_int8:>6.2f} TOPS)")

        configs = int4_kernel_configs if int4_kernel_configs else [None]
        best = None
        for kernel_config in configs:
            def fn_int4():
                kwargs = {}
                if kernel_config is not None:
                    kwargs["int4_kernel_config"] = kernel_config
                return sageattn_qk_int4_pv_fp8_cuda(
                    q, k, v,
                    tensor_layout="HND",
                    is_causal=args.is_causal,
                    qk_quant_gran="per_thread",
                    pv_accum_dtype="fp32+fp16",
                    **kwargs,
                )

            ms_int4 = bench(fn_int4, num_warmups=args.num_warmups, num_tests=args.num_tests, high_precision=False)
            out_int4 = fn_int4()
            diff = (out_int8 - out_int4).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            tops_int4 = tops(args.batch_size, args.num_heads, args.head_dim, seqlen, ms_int4, args.is_causal)
            speedup = ms_int8 / ms_int4
            label = "auto" if kernel_config is None else f"cfg{kernel_config}"
            print(
                f"           {label:>4s} -> "
                f"{ms_int4:>7.3f} ms ({tops_int4:>6.2f} TOPS) | "
                f"speedup={speedup:>5.3f}x | "
                f"mean_abs_diff={mean_diff:.5f}, max_abs_diff={max_diff:.5f}"
            )
            if best is None or ms_int4 < best[0]:
                best = (ms_int4, label)

        if len(configs) > 1 and best is not None:
            print(f"           best -> {best[1]} @ {best[0]:.3f} ms")


if __name__ == "__main__":
    main()
