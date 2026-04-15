from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def load_bench_module():
    path = Path(__file__).resolve().parents[1] / "bench" / "bench_qk_int4_pv_fp8_cuda_smooth_q.py"
    spec = spec_from_file_location("bench_qk_int4_pv_fp8_cuda_smooth_q", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_cases_parses_multiple_shapes():
    module = load_bench_module()

    cases = module.parse_cases("fp16,0,1,8,512,64;bf16,1,2,16,1024,128")

    assert cases == [
        ("fp16", False, 1, 8, 512, 64),
        ("bf16", True, 2, 16, 1024, 128),
    ]


def test_block_mean_q_repeats_each_block_mean():
    module = load_bench_module()

    q = torch.tensor(
        [[[[1.0], [3.0], [10.0], [14.0]]]],
        dtype=torch.float32,
    )

    out = module.block_mean_q(q, blkq=2)

    expected = torch.tensor(
        [[[[2.0], [2.0], [12.0], [12.0]]]],
        dtype=torch.float32,
    )
    assert torch.equal(out, expected)


def test_attention_scores_to_output_chunked_matches_full():
    module = load_bench_module()

    torch.manual_seed(0)
    q = torch.randn(1, 2, 5, 4, dtype=torch.float32)
    k = torch.randn(1, 2, 5, 4, dtype=torch.float32)
    v = torch.randn(1, 2, 5, 4, dtype=torch.float32)
    delta = torch.randn(1, 2, 5, 5, dtype=torch.float32) * 0.01
    sm_scale = q.size(-1) ** -0.5

    full_scores = (torch.matmul(q, k.transpose(-1, -2)) + delta) * sm_scale
    full_out = torch.matmul(torch.softmax(full_scores, dim=-1), v)

    chunked_out = module.attention_scores_to_output_chunked(
        q,
        k,
        v,
        sm_scale=sm_scale,
        delta=delta,
        is_causal=False,
        q_chunk_size=2,
    )

    assert torch.allclose(chunked_out, full_out, atol=1e-6, rtol=1e-6)
