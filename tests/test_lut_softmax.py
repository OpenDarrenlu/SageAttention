import torch

from sageattention.lut_softmax import lut_attention, lut_softmax


def test_lut_softmax_uint8_row_sums_are_close_to_255():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 16, dtype=torch.float32)
    probs_u8 = lut_softmax(logits, return_uint8=True)
    row_sums = probs_u8.to(torch.int32).sum(dim=-1)
    assert row_sums.min().item() >= 240
    assert row_sums.max().item() <= 270


def test_lut_attention_tracks_softmax_attention_reasonably():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 32, 64, dtype=torch.float32)
    k = torch.randn(1, 2, 32, 64, dtype=torch.float32)
    v = torch.randn(1, 2, 32, 64, dtype=torch.float32)

    ref_scores = torch.matmul(q, k.transpose(-1, -2)) * (64 ** -0.5)
    ref_probs = torch.softmax(ref_scores, dim=-1)
    ref_out = torch.matmul(ref_probs, v)

    approx_out = lut_attention(q, k, v, clip_value=4.5, num_entries=32)
    diff = (ref_out - approx_out).abs()
    assert diff.mean().item() < 0.08
    assert diff.max().item() < 0.6


def test_lut_attention_chunked_matches_non_chunked():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 64, 64, dtype=torch.float32)
    k = torch.randn(1, 2, 64, 64, dtype=torch.float32)
    v = torch.randn(1, 2, 64, 64, dtype=torch.float32)

    out_full = lut_attention(q, k, v, clip_value=4.5, num_entries=32)
    out_chunked = lut_attention(q, k, v, clip_value=4.5, num_entries=32, q_chunk_size=16)
    assert torch.allclose(out_full, out_chunked, atol=1e-6, rtol=1e-6)
