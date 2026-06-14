"""
SageAttention LUT P/V precision sweep on a dummy WanTransformer3DModel.

This script evaluates two levels of accuracy:

1. **Attention-level error**: for each transformer block we capture the
   inputs (Q, K, V) and output of the self-attention function, then compare
   the quantized output against the SDPA baseline on the *same* Q/K/V.
   This isolates the precision loss introduced by P/V quantization.

2. **Model-level error**: we compare the final output tensor of the whole
   Wan model when self-attention is replaced by each P/V precision combo,
   using the original SDPA output as the baseline.

Results are written to ``results/`` as JSON and CSV.
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

# Add repo root so ``import sageattention`` works when running from work4.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modify_wan4 import (
    WanAttnProcessor2_0,
    build_dummy_wan,
    make_sageattn_lut,
    set_sage_attn_wan,
)


P_DTYPES = ["fp16", "bf16", "int8", "int4", "mxfp8", "mxfp4", "nvfp4", "mxint4"]
V_DTYPES = ["int8", "int4", "int2"]

# Number of random seeds to average over for robustness.
NUM_SEEDS = 3
# Wan-like head dimension for the dummy model.
ATTENTION_HEAD_DIM = 128


class CapturingWanAttnProcessor(WanAttnProcessor2_0):
    """
    Wan processor that also records the (Q, K, V) and attention output of
    every self-attention call.
    """

    def __init__(self, attn_func, storage: List[Dict[str, torch.Tensor]]):
        super().__init__(attn_func)
        self.storage = storage

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # Only capture the main self-attention branch (image branch is tiny in
        # the dummy model and not representative).
        out = self.attn_func(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        self.storage.append({
            "q": query.detach().clone(),
            "k": key.detach().clone(),
            "v": value.detach().clone(),
            "out": out.detach().clone(),
        })

        hidden_states = out.transpose(1, 2).flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def set_capturing_processor(model, attn_func, storage):
    """Install a capturing processor on every transformer block."""
    for block in model.blocks:
        block.attn1.processor = CapturingWanAttnProcessor(attn_func, storage)


def build_inputs(model, in_channels: int = 16, num_frames: int = 4,
                 width: int = 64, height: int = 64, seed: int = 42):
    """Build deterministic random inputs for the dummy Wan model."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    torch.manual_seed(seed)
    x = torch.randn(1, in_channels, num_frames, width, height, dtype=dtype, device=device)
    encoder_hidden_states = torch.randn(1, 512 + in_channels, 4096, dtype=dtype, device=device)
    timestep = torch.randint(0, 1000, (1,), device=device).to(dtype)
    return x, encoder_hidden_states, timestep


def compute_error_metrics(pred: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """Return a dict of element-wise error metrics between pred and ref."""
    diff = (pred - ref).abs()
    ref_abs = ref.abs()
    ref_max = ref_abs.max()
    rel = diff / (ref_abs + 1e-6)
    return {
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "rel_max": rel.max().item(),
        "rel_mean": rel.mean().item(),
        "rmse": torch.sqrt((diff ** 2).mean()).item(),
        "cos_sim": torch.nn.functional.cosine_similarity(
            pred.flatten(), ref.flatten(), dim=0
        ).item(),
    }


def run_model_once(model, x, encoder_hidden_states, timestep):
    """Forward pass with no gradient."""
    with torch.no_grad():
        return model(hidden_states=x, encoder_hidden_states=encoder_hidden_states, timestep=timestep)[0]


def attention_level_experiment(model, seeds: List[int]) -> Dict[str, Dict[str, float]]:
    """
    Capture attention outputs for the SDPA baseline and each P/V combo over
    multiple random seeds, then average per-layer error metrics.
    """
    combo_to_results: Dict[str, List[Dict[str, float]]] = {}

    for seed in seeds:
        x, encoder_hidden_states, timestep = build_inputs(model, seed=seed)

        # Baseline capture
        baseline_storage: List[Dict[str, torch.Tensor]] = []
        set_capturing_processor(model, F.scaled_dot_product_attention, baseline_storage)
        _ = run_model_once(model, x, encoder_hidden_states, timestep)

        for p_dt in P_DTYPES:
            for v_dt in V_DTYPES:
                storage: List[Dict[str, torch.Tensor]] = []
                attn_func = make_sageattn_lut(p_dt, v_dt, p_block_n=64)
                set_capturing_processor(model, attn_func, storage)
                _ = run_model_once(model, x, encoder_hidden_states, timestep)

                if len(storage) != len(baseline_storage):
                    raise RuntimeError(f"Layer count mismatch for P={p_dt} V={v_dt}")

                layer_metrics = []
                for base, test in zip(baseline_storage, storage):
                    metrics = compute_error_metrics(test["out"].float(), base["out"].float())
                    layer_metrics.append(metrics)

                agg = {}
                for key in layer_metrics[0].keys():
                    vals = [m[key] for m in layer_metrics]
                    agg[f"{key}_mean"] = sum(vals) / len(vals)
                    agg[f"{key}_max"] = max(vals)

                combo = f"{p_dt}/{v_dt}"
                combo_to_results.setdefault(combo, []).append(agg)

    # Average across seeds
    averaged = {}
    for combo, seed_results in combo_to_results.items():
        averaged[combo] = {}
        for key in seed_results[0].keys():
            vals = [r[key] for r in seed_results]
            averaged[combo][key] = sum(vals) / len(vals)
    return averaged


def model_level_experiment(model, seeds: List[int]) -> Dict[str, Dict[str, float]]:
    """
    Run the full model for the SDPA baseline and each P/V combo over multiple
    seeds, then average final-output error metrics.
    """
    combo_to_results: Dict[str, List[Dict[str, float]]] = {}

    for seed in seeds:
        x, encoder_hidden_states, timestep = build_inputs(model, seed=seed)

        # Baseline
        set_sage_attn_wan(model, "fp16", "int8")  # placeholder
        for block in model.blocks:
            block.attn1.processor = WanAttnProcessor2_0(F.scaled_dot_product_attention)
        ref_out = run_model_once(model, x, encoder_hidden_states, timestep)

        for p_dt in P_DTYPES:
            for v_dt in V_DTYPES:
                set_sage_attn_wan(model, p_dt, v_dt)
                out = run_model_once(model, x, encoder_hidden_states, timestep)
                combo = f"{p_dt}/{v_dt}"
                combo_to_results.setdefault(combo, []).append(
                    compute_error_metrics(out.float(), ref_out.float())
                )

    averaged = {}
    for combo, seed_results in combo_to_results.items():
        averaged[combo] = {}
        for key in seed_results[0].keys():
            vals = [r[key] for r in seed_results]
            averaged[combo][key] = sum(vals) / len(vals)
    return averaged


def block_scaling_experiment(model, seeds: List[int]) -> Dict[str, Dict[str, float]]:
    """
    Focused experiment on the effect of P block-scaling granularity for the
    low-bit P formats that showed the largest attention-level error.
    V is fixed to int8 because it already gives the best accuracy.
    """
    p_dtypes = ["int4", "mxfp4", "nvfp4"]
    p_block_sizes = [64, 32, 16]
    combo_to_results: Dict[str, List[Dict[str, float]]] = {}

    for seed in seeds:
        x, encoder_hidden_states, timestep = build_inputs(model, seed=seed)

        # Baseline capture
        baseline_storage: List[Dict[str, torch.Tensor]] = []
        set_capturing_processor(model, F.scaled_dot_product_attention, baseline_storage)
        _ = run_model_once(model, x, encoder_hidden_states, timestep)

        for p_dt in p_dtypes:
            for pbn in p_block_sizes:
                storage: List[Dict[str, torch.Tensor]] = []
                attn_func = make_sageattn_lut(p_dt, "int8", p_block_n=pbn)
                set_capturing_processor(model, attn_func, storage)
                _ = run_model_once(model, x, encoder_hidden_states, timestep)

                layer_metrics = []
                for base, test in zip(baseline_storage, storage):
                    metrics = compute_error_metrics(test["out"].float(), base["out"].float())
                    layer_metrics.append(metrics)

                agg = {}
                for key in layer_metrics[0].keys():
                    vals = [m[key] for m in layer_metrics]
                    agg[f"{key}_mean"] = sum(vals) / len(vals)
                    agg[f"{key}_max"] = max(vals)

                combo = f"{p_dt}_b{pbn}/int8"
                combo_to_results.setdefault(combo, []).append(agg)

    averaged = {}
    for combo, seed_results in combo_to_results.items():
        averaged[combo] = {}
        for key in seed_results[0].keys():
            vals = [r[key] for r in seed_results]
            averaged[combo][key] = sum(vals) / len(vals)
    return averaged


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    seeds = list(range(NUM_SEEDS))

    # Use a slightly larger dummy model for more representative attention stats
    # while keeping the sweep fast.  head_dim=128 matches Wan.
    model = build_dummy_wan(
        in_channels=16,
        out_channels=16,
        num_layers=4,
        attention_head_dim=ATTENTION_HEAD_DIM,
        num_attention_heads=12,
        device=device,
    )

    print("=" * 80)
    print(f"Running attention-level precision sweep ({NUM_SEEDS} seeds, head_dim={ATTENTION_HEAD_DIM}) ...")
    print("=" * 80)
    t0 = time.time()
    attn_results = attention_level_experiment(model, seeds)
    print(f"Attention-level sweep finished in {time.time() - t0:.1f}s")

    print("\n" + "=" * 80)
    print(f"Running model-level precision sweep ({NUM_SEEDS} seeds) ...")
    print("=" * 80)
    t0 = time.time()
    model_results = model_level_experiment(model, seeds)
    print(f"Model-level sweep finished in {time.time() - t0:.1f}s")

    print("\n" + "=" * 80)
    print("Running block-scaling refinement experiment (low-bit P, V=int8) ...")
    print("=" * 80)
    t0 = time.time()
    block_results = block_scaling_experiment(model, seeds)
    print(f"Block-scaling sweep finished in {time.time() - t0:.1f}s")

    # Save raw results
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({
            "attention_level": attn_results,
            "model_level": model_results,
            "block_scaling": block_results,
        }, f, indent=2)

    # Save CSV tables
    _save_csv(os.path.join(out_dir, "attention_level.csv"), attn_results)
    _save_csv(os.path.join(out_dir, "model_level.csv"), model_results)
    _save_csv(os.path.join(out_dir, "block_scaling.csv"), block_results)

    # Print summary tables
    print("\n" + "=" * 80)
    print("Attention-level summary (mean across layers)")
    print("=" * 80)
    _print_table(attn_results, keys=["max_abs_err_mean", "mean_abs_err_mean", "rel_mean_mean", "cos_sim_mean"])

    print("\n" + "=" * 80)
    print("Model-level summary")
    print("=" * 80)
    _print_table(model_results, keys=["max_abs_err", "mean_abs_err", "rel_mean", "cos_sim"])

    print("\n" + "=" * 80)
    print("Block-scaling summary (low-bit P, V=int8, mean across layers)")
    print("=" * 80)
    _print_table(block_results, keys=["max_abs_err_mean", "mean_abs_err_mean", "rel_mean_mean", "cos_sim_mean"])

    # Recommend best combos
    _print_recommendations(attn_results, model_results)

    # Recommend block-scaled variants
    _print_block_scaling_recommendations(block_results)


def _save_csv(path: str, results: Dict[str, Dict[str, float]]):
    """Write a results dict to CSV."""
    if not results:
        return
    keys = list(next(iter(results.values())).keys())
    with open(path, "w") as f:
        f.write("combo," + ",".join(keys) + "\n")
        for combo, vals in results.items():
            f.write(combo + "," + ",".join(f"{vals[k]:.6e}" for k in keys) + "\n")


def _print_table(results: Dict[str, Dict[str, float]], keys: List[str]):
    """Pretty-print a results table."""
    header = f"{'P/V':<14}" + "".join(f"{k:<18}" for k in keys)
    print(header)
    print("-" * len(header))
    for combo in sorted(results.keys()):
        vals = results[combo]
        row = f"{combo:<14}" + "".join(f"{vals[k]:<18.6e}" for k in keys)
        print(row)


def _print_recommendations(attn_results: Dict[str, Dict[str, float]],
                           model_results: Dict[str, Dict[str, float]]):
    """
    Print a shortlist of P/V combos that achieve the best accuracy / bitrate
    trade-off.  We rank by attention-level mean absolute error first, then
    penalize low-bit P and V formats.
    """
    print("\n" + "=" * 80)
    print("Recommended precision combinations")
    print("=" * 80)

    # Bit-width estimate for ranking (P, V) -> total bits per element pair.
    p_bits = {"fp16": 16, "bf16": 16, "int8": 8, "int4": 4, "mxfp8": 8, "mxfp4": 4, "nvfp4": 4, "mxint4": 4}
    v_bits = {"int8": 8, "int4": 4, "int2": 2}

    ranked = []
    for combo, vals in attn_results.items():
        p_dt, v_dt = combo.split("/")
        mae = vals["mean_abs_err_mean"]
        bits = p_bits[p_dt] + v_bits[v_dt]
        # Simple score: lower MAE is better, lower bit-width is better.
        score = mae * bits
        ranked.append((combo, mae, bits, model_results[combo]["mean_abs_err"], score))

    ranked.sort(key=lambda x: x[1])  # sort by MAE
    print(f"{'Rank':<6}{'P/V':<14}{'Attn MAE':<14}{'Model MAE':<14}{'Bits(P+V)':<12}{'Score':<12}")
    print("-" * 72)
    for rank, (combo, mae, bits, model_mae, score) in enumerate(ranked[:10], 1):
        print(f"{rank:<6}{combo:<14}{mae:<14.6e}{model_mae:<14.6e}{bits:<12d}{score:<12.6e}")


def _print_block_scaling_recommendations(block_results: Dict[str, Dict[str, float]]):
    """
    Print the best block-scaled low-bit P configs and compare them to the
    default per-tile granularity.
    """
    print("\n" + "=" * 80)
    print("Block-scaling recommendations (P format_b{block_size} / V=int8)")
    print("=" * 80)

    ranked = []
    for combo, vals in block_results.items():
        mae = vals["mean_abs_err_mean"]
        # Extract bit-width: e.g. "int4_b32" -> P is 4 bits
        p_part = combo.split("/")[0]
        p_bits = 4  # all low-bit formats in this experiment are 4-bit
        bits = p_bits + 8  # V is int8
        score = mae * bits
        ranked.append((combo, mae, bits, score))

    ranked.sort(key=lambda x: x[1])
    print(f"{'Rank':<6}{'Config':<18}{'Attn MAE':<14}{'Bits(P+V)':<12}{'Score':<12}")
    print("-" * 62)
    for rank, (combo, mae, bits, score) in enumerate(ranked, 1):
        print(f"{rank:<6}{combo:<18}{mae:<14.6e}{bits:<12d}{score:<12.6e}")


if __name__ == "__main__":
    main()
