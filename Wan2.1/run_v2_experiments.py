#!/usr/bin/env python3
"""
Generate Wan2.1 videos for P=4bit/16bit + V=int2 (per-channel vs block-scaled).
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

# Configs the user cares about: P 4-bit or higher, V 2-bit
CONFIGS = [
    ("fp16", "int2", 0),
    ("fp16", "int2", 64),
    ("int8", "int2", 64),
    ("int4", "int2", 0),
    ("int4", "int2", 64),
    ("mxfp4", "int2", 64),
    ("nvfp4", "int2", 64),
    ("mxint4", "int2", 64),
]

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


def run_one(idx, gpu_id, p_dt, v_dt, v_bs):
    env = BASE_ENV.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tag = f"P{p_dt}_V{v_dt}_vb{v_bs}"
    cmd = BASE_CMD + [
        "--p_quant_dtype", p_dt,
        "--v_quant_dtype", v_dt,
        "--p_block_n", "64",
        "--v_block_size", str(v_bs),
    ]
    log_file = f"logs_v2/{tag}.log"
    os.makedirs("logs_v2", exist_ok=True)
    start = datetime.now()
    print(f"[{idx}/{len(CONFIGS)}] GPU={gpu_id} {tag}  START")
    with open(log_file, "w") as f:
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
    print(f"[{idx}/{len(CONFIGS)}] GPU={gpu_id} {tag}  {status}  duration={duration:.1f}s")
    return idx, gpu_id, tag, ret, duration


def main():
    print(f"Detected {NUM_GPUS} GPUs, running {len(CONFIGS)} V=int2 jobs...")
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = [executor.submit(run_one, i, i % NUM_GPUS, *cfg) for i, cfg in enumerate(CONFIGS)]
        results = [future.result() for future in as_completed(futures)]

    print("\n" + "=" * 80)
    print("All jobs finished")
    print("=" * 80)
    ok = sum(1 for r in results if r[3] == 0)
    print(f"Success: {ok}/{len(results)}")
    for idx, gpu_id, tag, ret, duration in sorted(results):
        status = "OK" if ret == 0 else f"FAIL({ret})"
        print(f"  [{idx}] GPU{gpu_id} {tag:25s} {status:8s} {duration:8.1f}s")


if __name__ == "__main__":
    main()
