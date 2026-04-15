import argparse
import os
import statistics
import sys
from typing import List, Sequence, Tuple

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sageattention import sageattn_qk_int4_pv_fp8_cuda, sageattn_qk_int8_pv_fp8_cuda
from sageattention.core import get_int4_kernel_config
from sageattention.triton.quant_per_thread import per_thread_int4, unpack_int4


Case = Tuple[str, bool, int, int, int, int]


def parse_cases(raw: str) -> List[Case]:
    cases: List[Case] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        dtype, causal, batch, heads, seqlen, head_dim = [part.strip() for part in item.split(",")]
        cases.append((dtype, bool(int(causal)), int(batch), int(heads), int(seqlen), int(head_dim)))
    return cases


def block_mean_q(q: torch.Tensor, blkq: int) -> torch.Tensor:
    out = torch.empty_like(q)
    seq_len = q.size(2)
    for start in range(0, seq_len, blkq):
        end = min(start + blkq, seq_len)
        mean = q[:, :, start:end, :].mean(dim=2, keepdim=True)
        out[:, :, start:end, :] = mean
    return out


def q_scales_per_token(q_scale: torch.Tensor, seq_len: int, warpq: int) -> torch.Tensor:
    tokens = torch.arange(seq_len, device=q_scale.device)
    idx = (tokens // warpq) * 8 + (tokens % 8)
    return q_scale[:, :, idx.long()]


def k_scales_per_token(k_scale: torch.Tensor, seq_len: int, warpk: int) -> torch.Tensor:
    tokens = torch.arange(seq_len, device=k_scale.device)
    idx = (tokens // warpk) * 4 + ((tokens % 8) // 2)
    return k_scale[:, :, idx.long()]


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    seq_q, seq_k = scores.size(-2), scores.size(-1)
    mask = torch.triu(
        torch.ones((seq_q, seq_k), device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    return scores.masked_fill(mask.view(1, 1, seq_q, seq_k), float("-inf"))


def attention_scores_to_output_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    delta: torch.Tensor | None,
    is_causal: bool,
    q_chunk_size: int = 1024,
) -> torch.Tensor:
    seq_len = q.size(2)
    out = torch.empty_like(v)
    k_t = k.transpose(-1, -2)

    for start in range(0, seq_len, q_chunk_size):
        end = min(start + q_chunk_size, seq_len)
        scores = torch.matmul(q[:, :, start:end, :], k_t)
        if delta is not None:
            scores = scores + delta[:, :, start:end, :]
        scores = scores * sm_scale
        if is_causal:
            chunk_q = end - start
            row_ids = torch.arange(start, end, device=scores.device).view(1, 1, chunk_q, 1)
            col_ids = torch.arange(seq_len, device=scores.device).view(1, 1, 1, seq_len)
            scores = scores.masked_fill(col_ids > row_ids, float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out[:, :, start:end, :] = torch.matmul(probs, v)
    return out


def emulate_int4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    smooth_q: bool,
    int4_kernel_config: int | None = None,
) -> torch.Tensor:
    batch, heads, seq_len, head_dim = q.shape
    sm_scale = head_dim ** -0.5
    _, blkq, warpq, blkk, warpk = get_int4_kernel_config(
        head_dim,
        seq_len,
        seq_len,
        int4_kernel_config,
    )

    km = k.mean(dim=2, keepdim=True)
    gamma_k = k - km

    if smooth_q:
        q_bar = block_mean_q(q, blkq)
        gamma_q = q - q_bar
        q_int4, q_scale, k_int4, k_scale = per_thread_int4(
            gamma_q,
            gamma_k,
            km=None,
            tensor_layout="HND",
            BLKQ=blkq,
            WARPQ=warpq,
            BLKK=blkk,
            WARPK=warpk,
        )
        q_hat = unpack_int4(q_int4).float() * q_scales_per_token(q_scale, seq_len, warpq).unsqueeze(-1)
        k_hat = unpack_int4(k_int4).float() * k_scales_per_token(k_scale, seq_len, warpk).unsqueeze(-1)
        delta_s = torch.matmul(q_bar.float(), gamma_k.float().transpose(-1, -2))
    else:
        q_int4, q_scale, k_int4, k_scale = per_thread_int4(
            q,
            k,
            km=km,
            tensor_layout="HND",
            BLKQ=blkq,
            WARPQ=warpq,
            BLKK=blkk,
            WARPK=warpk,
        )
        q_hat = unpack_int4(q_int4).float() * q_scales_per_token(q_scale, seq_len, warpq).unsqueeze(-1)
        k_hat = unpack_int4(k_int4).float() * k_scales_per_token(k_scale, seq_len, warpk).unsqueeze(-1)
        delta_s = None

    return attention_scores_to_output_chunked(
        q_hat,
        k_hat,
        v,
        sm_scale=sm_scale,
        delta=delta_s,
        is_causal=is_causal,
    )


def summarize(values: Sequence[float]) -> Tuple[float, float]:
    avg = sum(values) / len(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return avg, std


def run_case(
    dtype_name: str,
    is_causal: bool,
    batch: int,
    heads: int,
    seqlen: int,
    head_dim: int,
    seeds: Sequence[int],
    int4_kernel_config: int | None,
) -> dict:
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    actual_means: List[float] = []
    actual_maxes: List[float] = []
    smooth_means: List[float] = []
    smooth_maxes: List[float] = []

    for seed in seeds:
        torch.manual_seed(seed)
        q = torch.randn(batch, heads, seqlen, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, heads, seqlen, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch, heads, seqlen, head_dim, dtype=dtype, device="cuda")

        out_int8 = sageattn_qk_int8_pv_fp8_cuda(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=is_causal,
            qk_quant_gran="per_thread",
            pv_accum_dtype="fp32+fp16",
        )
        out_actual = sageattn_qk_int4_pv_fp8_cuda(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=is_causal,
            qk_quant_gran="per_thread",
            pv_accum_dtype="fp32+fp16",
            int4_kernel_config=int4_kernel_config,
        )
        out_smooth_q = emulate_int4_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            smooth_q=True,
            int4_kernel_config=int4_kernel_config,
        )

        actual_diff = (out_int8 - out_actual).abs().float()
        smooth_diff = (out_int8 - out_smooth_q).abs().float()
        actual_means.append(actual_diff.mean().item())
        actual_maxes.append(actual_diff.max().item())
        smooth_means.append(smooth_diff.mean().item())
        smooth_maxes.append(smooth_diff.max().item())

    actual_mean_avg, actual_mean_std = summarize(actual_means)
    smooth_mean_avg, smooth_mean_std = summarize(smooth_means)
    return {
        "dtype": dtype_name,
        "causal": is_causal,
        "batch": batch,
        "heads": heads,
        "seqlen": seqlen,
        "head_dim": head_dim,
        "actual_mean_avg": actual_mean_avg,
        "actual_mean_std": actual_mean_std,
        "actual_max_max": max(actual_maxes),
        "smooth_mean_avg": smooth_mean_avg,
        "smooth_mean_std": smooth_mean_std,
        "smooth_max_max": max(smooth_maxes),
        "delta_mean": smooth_mean_avg - actual_mean_avg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare int4 smooth-q emulation against the current int4 branch and int8 baseline")
    parser.add_argument(
        "--cases",
        type=str,
        default="fp16,0,1,8,512,64;fp16,0,2,16,1024,128;fp16,1,2,16,1024,128;bf16,0,2,16,1024,128;bf16,1,2,16,1024,128",
        help="Semicolon-separated cases: dtype,causal,batch,heads,seqlen,head_dim",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--int4-kernel-config", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 9):
        raise SystemExit(f"This benchmark targets sm89 (RTX 4090 class), got sm{capability[0]}{capability[1]}")

    cases = parse_cases(args.cases)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    print(
        "dtype,causal,B,H,L,D,"
        "actual_mean_avg,actual_mean_std,actual_max_max,"
        "smoothq_mean_avg,smoothq_mean_std,smoothq_max_max,delta_mean"
    )
    for dtype_name, is_causal, batch, heads, seqlen, head_dim in cases:
        row = run_case(
            dtype_name,
            is_causal,
            batch,
            heads,
            seqlen,
            head_dim,
            seeds,
            args.int4_kernel_config,
        )
        print(
            f"{row['dtype']},{int(row['causal'])},{row['batch']},{row['heads']},{row['seqlen']},{row['head_dim']},"
            f"{row['actual_mean_avg']:.6f},{row['actual_mean_std']:.6f},{row['actual_max_max']:.6f},"
            f"{row['smooth_mean_avg']:.6f},{row['smooth_mean_std']:.6f},{row['smooth_max_max']:.6f},"
            f"{row['delta_mean']:+.6f}"
        )


if __name__ == "__main__":
    main()
