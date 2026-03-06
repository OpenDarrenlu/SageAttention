from __future__ import annotations

from typing import Optional

import torch


def build_uint8_exp_lut(
    num_entries: int = 32,
    clip_value: float = 4.5,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if num_entries < 2:
        raise ValueError("num_entries must be at least 2")
    if clip_value <= 0:
        raise ValueError("clip_value must be positive")

    xs = torch.linspace(0.0, clip_value, num_entries, dtype=torch.float32, device=device)
    lut = torch.exp(-xs)
    lut = torch.round(lut * 255.0).clamp_(0, 255).to(torch.uint8)
    return lut


def _rounded_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.div(
        numerator + torch.div(denominator, 2, rounding_mode="floor"),
        denominator,
        rounding_mode="floor",
    )


def lut_softmax(
    logits: torch.Tensor,
    *,
    clip_value: float = 4.5,
    num_entries: int = 32,
    return_uint8: bool = False,
) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must have at least 2 dimensions")

    logits_fp32 = logits.to(torch.float32)
    row_max = logits_fp32.max(dim=-1, keepdim=True).values
    distances = (row_max - logits_fp32).clamp_(min=0.0, max=clip_value)

    scale = (num_entries - 1) / clip_value
    indices = torch.round(distances * scale).clamp_(0, num_entries - 1).to(torch.long)

    lut = build_uint8_exp_lut(num_entries=num_entries, clip_value=clip_value, device=logits.device)
    exp_u8 = lut[indices]
    exp_i32 = exp_u8.to(torch.int32)
    denom = exp_i32.sum(dim=-1, keepdim=True).clamp_min_(1)
    probs_i32 = exp_i32.mul_(255)
    probs_i32 = _rounded_div(probs_i32, denom).clamp_(0, 255)

    row_sum = probs_i32.sum(dim=-1, keepdim=True)
    correction = 255 - row_sum
    row_argmax = logits_fp32.argmax(dim=-1, keepdim=True)
    probs_i32.scatter_add_(-1, row_argmax, correction)
    probs_u8 = probs_i32.clamp_(0, 255).to(torch.uint8)

    if return_uint8:
        return probs_u8
    return probs_u8.to(logits_fp32.dtype) / 255.0


def lut_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sm_scale: Optional[float] = None,
    is_causal: bool = False,
    clip_value: float = 4.5,
    num_entries: int = 32,
    q_chunk_size: Optional[int] = None,
    return_prob_uint8: bool = False,
):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 tensors in HND layout")
    if q.shape[:-1] != k.shape[:-1] or k.shape[:-1] != v.shape[:-1] or q.shape[-1] != k.shape[-1]:
        raise ValueError("This experimental path currently expects matching HND shapes for q, k, v")

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5

    if q_chunk_size is None or q_chunk_size >= q.size(-2):
        scores = torch.matmul(q, k.transpose(-1, -2)).to(torch.float32) * sm_scale
        if is_causal:
            seq_q, seq_k = scores.size(-2), scores.size(-1)
            mask = torch.triu(torch.ones((seq_q, seq_k), device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.view(1, 1, seq_q, seq_k), float("-inf"))

        probs_u8 = lut_softmax(scores, clip_value=clip_value, num_entries=num_entries, return_uint8=True)
        probs = probs_u8.to(torch.float32) / 255.0
        out = torch.matmul(probs.to(v.dtype), v)
    else:
        seq_q = q.size(-2)
        seq_k = k.size(-2)
        key_t = k.transpose(-1, -2)
        out_chunks = []
        prob_chunks = [] if return_prob_uint8 else None
        key_positions = torch.arange(seq_k, device=q.device)

        for start in range(0, seq_q, q_chunk_size):
            end = min(start + q_chunk_size, seq_q)
            q_chunk = q[:, :, start:end, :]
            scores = torch.matmul(q_chunk, key_t).to(torch.float32) * sm_scale

            if is_causal:
                query_positions = torch.arange(start, end, device=q.device)
                mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                scores = scores.masked_fill(mask.view(1, 1, end - start, seq_k), float("-inf"))

            probs_u8_chunk = lut_softmax(scores, clip_value=clip_value, num_entries=num_entries, return_uint8=True)
            probs_chunk = probs_u8_chunk.to(torch.float32) / 255.0
            out_chunks.append(torch.matmul(probs_chunk.to(v.dtype), v))
            if prob_chunks is not None:
                prob_chunks.append(probs_u8_chunk)

        out = torch.cat(out_chunks, dim=2)
        probs_u8 = torch.cat(prob_chunks, dim=2) if prob_chunks is not None else None

    if return_prob_uint8:
        return out, probs_u8
    return out
