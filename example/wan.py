from diffusers import WanPipeline
from diffusers.utils import export_to_video
import torch, os, gc
import torch.nn.functional as F
import os
import argparse
from diffusers.utils import export_to_video
from modify_model.modify_wan import set_sage_attn_wan
from tqdm import tqdm
from sageattention import sageattn, sageattn_pint

def normal_attn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    # clamp attn_weights to [0, fp16.tiny]
    tiny_P = torch.finfo(attn_weights.dtype).tiny
    # attn_weights = attn_weights.clamp(min=0, max=f16_tiny)
    attn_weights = attn_weights.clamp(min=tiny_P, max=1)
    return torch.matmul(attn_weights, value)

ATTNENTION = {
    "sage": sageattn,
    "sdpa": F.scaled_dot_product_attention,
    "sage_pint": sageattn_pint,
}

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="THUDM/CogVideoX1.5-5B", help='Model path')
parser.add_argument('--compile', action='store_true', help='Compile the model')
parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'sage', 'sage_pint'], help='Attention type')
args = parser.parse_args()



prompt = "A serene night scene in a forested area. The first frame shows a tranquil lake reflecting the star-filled sky above. The second frame reveals a beautiful sunset, casting a warm glow over the landscape. The third frame showcases the night sky, filled with stars and a vibrant Milky Way galaxy. The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. The style of the video is naturalistic, emphasizing the beauty of the night sky and the peacefulness of the forest."

torch.manual_seed(42)

# Available models: Wan-AI/Wan2.1-I2V-14B-720P-Diffusers or Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
# model_id = "../../Wan2.1-T2V-1.3B-Diffusers" # query shape: torch.Size([1, 12, 14040, 128]), key shape: torch.Size([1, 12, 14040, 128]), value shape: torch.Size([1, 12, 14040, 128])
model_id = args.model_path

pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload()

if args.compile:
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

set_sage_attn_wan(pipe.transformer, ATTNENTION[args.attention_type])
# set_sage_attn_wan(pipe.transformer, normal_attn)

with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
    video = pipe(prompt=prompt, negative_prompt=None, num_frames=33).frames[0]
export_to_video(video, f"wan_{args.attention_type}.mp4", fps=16)
del video
gc.collect()
torch.cuda.empty_cache()
