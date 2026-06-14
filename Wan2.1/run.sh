# Run Wan2.1 T2V-1.3B with SageAttention LUT and configurable P/V precision.
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash run.sh
#
# Switch precision by setting environment variables, e.g.:
#   P_DTYPE=int8 V_DTYPE=int8 P_BLOCK_N=64 CUDA_VISIBLE_DEVICES=2 bash run.sh
#   P_DTYPE=nvfp4 V_DTYPE=int8 P_BLOCK_N=16 CUDA_VISIBLE_DEVICES=2 bash run.sh
#   P_DTYPE=int4 V_DTYPE=int8 P_BLOCK_N=16 CUDA_VISIBLE_DEVICES=2 bash run.sh

# Fix torch.linalg / MKL symbol issue in this environment
export MKL_THREADING_LAYER=GNU
export LD_PRELOAD=/opt/conda/lib/libmkl_rt.so${LD_PRELOAD:+:$LD_PRELOAD}

P_DTYPE=${P_DTYPE:-nvfp4}
V_DTYPE=${V_DTYPE:-int8}
P_BLOCK_N=${P_BLOCK_N:-16}

# 5-second video @ 16 fps => 81 frames (must be 4n+1)
FRAME_NUM=81

CUDA_VISIBLE_DEVICES=2 python generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --frame_num ${FRAME_NUM} \
  --ckpt_dir /home/lutingzhan/workspace_infer/Wan2.1-T2V-1.3B \
  --offload_model True \
  --t5_cpu \
  --use_lut \
  --p_quant_dtype ${P_DTYPE} \
  --v_quant_dtype ${V_DTYPE} \
  --p_block_n ${P_BLOCK_N} \
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
