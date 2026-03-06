import pytest
import torch

pytest.importorskip("triton")

from sageattention.core import get_int4_kernel_config
from sageattention.triton.quant_per_thread import pack_int4, unpack_int4


def require_sm89_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability(0) != (8, 9):
        pytest.skip("This test targets sm89 (RTX 4090 class)")


def make_qkv(tensor_layout: str, seqlen: int = 256, head_dim: int = 128):
    if tensor_layout == "HND":
        shape = (2, 8, seqlen, head_dim)
    else:
        shape = (2, seqlen, 8, head_dim)
    q = torch.randn(*shape, dtype=torch.float16, device="cuda")
    k = torch.randn(*shape, dtype=torch.float16, device="cuda")
    v = torch.randn(*shape, dtype=torch.float16, device="cuda")
    return q, k, v


def test_pack_unpack_int4_roundtrip():
    x = torch.randint(-8, 8, (2, 3, 4, 128), dtype=torch.int8)
    packed = pack_int4(x)
    restored = unpack_int4(packed)
    assert packed.shape == (2, 3, 4, 64)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)


def test_get_int4_kernel_config_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("SAGEATTN_INT4_KERNEL_CONFIG", raising=False)
    assert get_int4_kernel_config(head_dim=128, qo_len=2048, kv_len=2048) == (0, 128, 32, 64, 64)
    assert get_int4_kernel_config(head_dim=128, qo_len=2048, kv_len=2048, kernel_config=1) == (1, 128, 64, 64, 64)

    monkeypatch.setenv("SAGEATTN_INT4_KERNEL_CONFIG", "2")
    assert get_int4_kernel_config(head_dim=128, qo_len=2048, kv_len=2048) == (2, 64, 64, 64, 64)

    monkeypatch.setenv("SAGEATTN_INT4_KERNEL_CONFIG", "bad")
    with pytest.raises(ValueError, match="Invalid SAGEATTN_INT4_KERNEL_CONFIG"):
        get_int4_kernel_config(head_dim=128, qo_len=2048, kv_len=2048)


@pytest.mark.cuda
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_sageattn_int4_matches_int8_cuda(tensor_layout):
    require_sm89_cuda()
    from sageattention.core import (
        sageattn_qk_int4_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
    )

    torch.manual_seed(0)
    q, k, v = make_qkv(tensor_layout=tensor_layout)

    o_int8 = sageattn_qk_int8_pv_fp8_cuda(
        q, k, v,
        tensor_layout=tensor_layout,
        is_causal=False,
        qk_quant_gran="per_thread",
        pv_accum_dtype="fp32+fp16",
    )
    o_int4 = sageattn_qk_int4_pv_fp8_cuda(
        q, k, v,
        tensor_layout=tensor_layout,
        is_causal=False,
        qk_quant_gran="per_thread",
        pv_accum_dtype="fp32+fp16",
    )

    assert torch.isfinite(o_int4).all()
    max_diff = (o_int8 - o_int4).abs().max().item()
    mean_diff = (o_int8 - o_int4).abs().mean().item()
    assert max_diff < 0.2
    assert mean_diff < 0.02


@pytest.mark.cuda
def test_sageattn_top_level_int4_dispatch_matches_direct():
    require_sm89_cuda()

    from sageattention import sageattn, sageattn_qk_int4_pv_fp8_cuda

    torch.manual_seed(0)
    q, k, v = make_qkv(tensor_layout="HND", seqlen=128)

    out_top, lse_top = sageattn(
        q, k, v,
        tensor_layout="HND",
        qk_quant_dtype="int4",
        return_lse=True,
    )
    out_direct, lse_direct = sageattn_qk_int4_pv_fp8_cuda(
        q, k, v,
        tensor_layout="HND",
        return_lse=True,
        pv_accum_dtype="fp32+fp16",
    )

    assert torch.equal(out_top, out_direct)
    assert torch.equal(lse_top, lse_direct)
    assert lse_top.shape == (q.size(0), q.size(1), q.size(2))
    assert torch.isfinite(lse_top).all()


@pytest.mark.cuda
def test_sageattn_rejects_int4_on_unsupported_arch(monkeypatch):
    require_sm89_cuda()

    from sageattention import sageattn
    import sageattention.core as sageattention_core

    q, k, v = make_qkv(tensor_layout="HND", seqlen=64)
    monkeypatch.setattr(sageattention_core, "get_cuda_arch_versions", lambda: ["sm90"])

    with pytest.raises(ValueError, match="Unsupported qk_quant_dtype for sm90: int4"):
        sageattn(q, k, v, tensor_layout="HND", qk_quant_dtype="int4")
