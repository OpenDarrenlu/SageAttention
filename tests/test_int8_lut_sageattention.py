import pytest
import torch

pytest.importorskip("triton")


@pytest.mark.parametrize(
    ("is_causal", "mean_threshold", "max_threshold"),
    [
        (False, 0.08, 1.0),
        (True, 0.12, 4.0),
    ],
)
@pytest.mark.cuda
def test_sageattn_int8_lut_matches_int8_cuda_reasonably(is_causal, mean_threshold, max_threshold):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability(0) != (8, 9):
        pytest.skip("This test targets sm89 (RTX 4090 class)")

    from sageattention import sageattn_qk_int8_pv_fp8_cuda, sageattn_qk_int8_pv_fp8_cuda_lut

    torch.manual_seed(0)
    q = torch.randn(1, 8, 256, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 8, 256, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 8, 256, 128, dtype=torch.float16, device="cuda")

    out_ref = sageattn_qk_int8_pv_fp8_cuda(
        q, k, v,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran="per_thread",
        pv_accum_dtype="fp32+fp16",
    )
    out_lut = sageattn_qk_int8_pv_fp8_cuda_lut(
        q, k, v,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran="per_thread",
        pv_accum_dtype="fp32+fp16",
    )

    assert torch.isfinite(out_lut).all()
    diff = (out_ref - out_lut).abs()
    assert diff.mean().item() < mean_threshold
    assert diff.max().item() < max_threshold
