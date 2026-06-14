#!/usr/bin/env python3
"""
Generate Wan2.1 T2V videos for all P/V precision combinations.

Runs 24 (P-dtype x V-dtype) jobs across all available GPUs, with exactly one
job per GPU at any time to avoid VRAM contention.  Each job uses the default
p_block_n=64 granularity.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    import torch
    NUM_GPUS = torch.cuda.device_count()
except Exception:
    NUM_GPUS = 8

P_DTYPES = ["fp16", "bf16", "int8", "int4", "mxfp8", "mxfp4", "nvfp4", "mxint4"]
V_DTYPES = ["int8", "int4", "int2"]

BASE_CMD = [
    sys.executable, "generate.py",
    "--task", "t2v-1.3B",
    "--size", "832*480",
    "--frame_num", "81",
    "--ckpt_dir", "/home/lutingzhan/workspace_infer/Wan2.1-T2V-1.3B",
    "--offload_model", "True",
    "--t5_cpu",
    "--use_lut",
    "--sample_shift", "8",
    "--sample_guide_scale", "6",
    "--prompt",
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
]

BASE_ENV = os.environ.copy()
BASE_ENV["MKL_THREADING_LAYER"] = "GNU"
BASE_ENV["LD_PRELOAD"] = "/opt/conda/lib/libmkl_rt.so" + (
    ":" + BASE_ENV["LD_PRELOAD"] if BASE_ENV.get("LD_PRELOAD") else ""
)


def make_jobs_for_gpu(gpu_id):
    """Return list of (global_idx, tag, cmd) assigned to a given GPU."""
    jobs = []
    total = len(P_DTYPES) * len(V_DTYPES)
    for idx in range(gpu_id, total, NUM_GPUS):
        p_dt = P_DTYPES[idx // len(V_DTYPES)]
        v_dt = V_DTYPES[idx % len(V_DTYPES)]
        tag = f"P{p_dt}_V{v_dt}_b64"
        cmd = BASE_CMD + [
            "--p_quant_dtype", p_dt,
            "--v_quant_dtype", v_dt,
            "--p_block_n", "64",
        ]
        jobs.append((idx, tag, cmd))
    return jobs


def run_jobs_on_gpu(gpu_id):
    """Run all jobs assigned to this GPU sequentially."""
    results = []
    jobs = make_jobs_for_gpu(gpu_id)
    env = BASE_ENV.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.makedirs("logs", exist_ok=True)

    for idx, tag, cmd in jobs:
        log_file = f"logs/job_{idx:02d}_{tag}.log"
        start = datetime.now()
        print(f"[{idx:02d}/{24}] GPU={gpu_id} {tag}  START")
        with open(log_file, "w") as f:
            f.write(f"CMD: {' '.join(cmd)}\n")
            f.write(f"CUDA_VISIBLE_DEVICES={gpu_id}\n")
            f.write(f"START: {start.isoformat()}\n")
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            ret = proc.wait()
        end = datetime.now()
        duration = (end - start).total_seconds()
        status = "OK" if ret == 0 else f"FAIL({ret})"
        print(f"[{idx:02d}/{24}] GPU={gpu_id} {tag}  {status}  duration={duration:.1f}s")
        results.append((idx, gpu_id, tag, ret, duration, log_file))
    return results


def main():
    print(f"Detected {NUM_GPUS} GPUs, scheduling 24 jobs (1 per GPU at a time)...")
    all_results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = [executor.submit(run_jobs_on_gpu, gpu_id) for gpu_id in range(NUM_GPUS)]
        for future in as_completed(futures):
            all_results.extend(future.result())

    print("\n" + "=" * 80)
    print("All jobs finished")
    print("=" * 80)
    ok = sum(1 for r in all_results if r[3] == 0)
    print(f"Success: {ok}/{len(all_results)}")
    for idx, gpu_id, tag, ret, duration, log_file in sorted(all_results):
        status = "OK" if ret == 0 else f"FAIL({ret})"
        print(f"  [{idx:02d}] GPU{gpu_id} {tag:25s} {status:8s} {duration:8.1f}s  log={log_file}")


if __name__ == "__main__":
    main()
