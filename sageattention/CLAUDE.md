# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
# Install from source (requires CUDA >= 12.0, torch >= 2.3.0, triton >= 3.0.0)
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32  # optional
python setup.py install

# Skip CUDA build (e.g., CI or pure-Triton work)
SAGEATTN_SKIP_CUDA_BUILD=1 python setup.py install

# Target specific architectures only
TORCH_CUDA_ARCH_LIST="8.9;9.0" python setup.py install
```

**Dependencies**: `torch>=2.3.0`, `triton>=3.0.0`, CUDA toolkit matching GPU arch:
- CUDA >= 12.0 for Ampere (SM80/SM86)
- CUDA >= 12.4 for Ada (SM89, FP8 support)
- CUDA >= 12.3 for Hopper (SM90)
- CUDA >= 12.8 for Blackwell (SM120)

## Architecture

This is **SageAttention** — a plug-and-play low-bit attention library that quantizes Q/K to INT8 and optionally P/V to FP8, achieving 2-3x speedup over FlashAttention with minimal accuracy loss.

### Module layering

```
sageattention/__init__.py     — public API (sageattn, sageattn_lut, sageattn_pint, etc.)
    ├── core.py               — main kernels + top-level dispatcher (auto GPU arch selection)
    ├── core_lut.py            — LUT-based attention variant (research branch)
    ├── core_pint.py           — PINT variant using Triton kernel
    ├── core_pint_torch.py     — PINT variant using torch ops + torchmm.matmul
    ├── core_Pcodebook.py      — P cookbook quantized attention (research branch)
    ├── quant.py               — CUDA quantization (per_block_int8, per_warp_int8, per_channel_fp8, sub_mean)
    ├── fa3_wrapper.py         — FlashAttention3 wrappers (fa3, fa3_fp8)
    ├── sm80_compile.py        — torch.library custom_ops → _qattn_sm80 compiled extension (Ampere)
    ├── sm89_compile.py        — torch.library custom_ops → _qattn_sm89 compiled extension (Ada)
    ├── sm90_compile.py        — torch.library custom_ops → _qattn_sm90 compiled extension (Hopper)
    ├── triton/                — Triton JIT kernels
    │   ├── quant_per_block.py / quant_per_thread.py / quant_per_channel.py / quant_pint.py
    │   ├── attn_qk_int8_per_block.py / attn_qk_int8_per_block_causal.py
    │   ├── attn_qk_int8_block_varlen.py (varlen) / attn_qk_int8_lut_v_int8.py
    │   └── attn_qk_int8_pint_v_int8.py
    ├── csrc/                  — CUDA/C++ source for compiled extensions
    │   ├── fused/             → _fused (quantization: per_block, per_warp, sub_mean, transpose_pad_permute, mean_scale_fuse_quant)
    │   └── qattn/             → _qattn_sm80 / _qattn_sm89 / _qattn_sm90 (attention compute kernels)
    └── _fused.cpython-*.so, _qattn_sm80.cpython-*.so, _qattn_sm89.cpython-*.so  — precompiled binaries
```

### Key design patterns

- **GPU arch auto-dispatch**: `sageattn()` reads `torch.cuda.get_device_capability()` and routes to the best available kernel (Triton for SM86, CUDA FP16 for SM80, CUDA FP8 for SM89/SM90/SM120). The `SM80_ENABLED`/`SM89_ENABLED`/`SM90_ENABLED` flags come from trying to import the corresponding `sm*_compile` modules.

- **Quantized attention pipeline**: (1) smooth K by subtracting per-sequence mean → (2) quantize Q/K to INT8 (per-block or per-thread granularity) → (3) INT8 QK matmul with scale corrections → (4) softmax → (5) quantize V (FP8 or INT8, per-channel) → (6) PV matmul with FP32 accumulation buffer. The `sm_scale` is multiplied by `1.44269504` (1/ln2) to convert softmax to base-2 in Triton kernels.

- **Two tensor layouts**: `"HND"` = `(B, heads, seq, dim)` and `"NHD"` = `(B, seq, heads, dim)`. Internally encoded as `_tensor_layout`: 0=NHD, 1=HND.

- **torch.library custom_ops**: All CUDA kernels are registered via `torch.library.custom_op` with `mutates_args=("output",)` so they work with `torch.compile`. Fake implementations return correctly-shaped empty LSE tensors.

- **Current branch (`research_ltz`)**: Contains experimental attention variants — `sageattn_lut` (LUT-based), `sageattn_pint`/`sageattn_pint_torch` (PINT, uses external `torchmm` package for INT32 block matmul), and `sageattn_pcodebook_torch` (P cookbook quantization). These are activated via `if True:` bypassing arch checks — work in progress.

### Building compiled extensions

The CUDA extensions are defined in `setup.py` and conditionally built based on GPU compute capability:
- `_fused` — always built (quantization kernels)
- `_qattn_sm80` — built for SM80/SM86/SM89/SM90 (Ampere+)
- `_qattn_sm89` — built for SM89/SM90 (Ada+, FP8 MMA)
- `_qattn_sm90` — built for SM90 only (Hopper, WGMMA)

The precompiled `.so` files should be deleted before install (e.g., `rm sageattention/_fused*.so sageattention/_qattn*.so`).
