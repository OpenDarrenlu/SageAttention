from fast_hadamard_transform import hadamard_transform
import torch
import math

tensor_path = "../qkv_tensors_30layers/qkv_tensors_0.pt"
qkv = torch.load(tensor_path, map_location='cuda')  # 直接加载到 GPU
q, k, v = qkv["query"], qkv["key"], qkv["value"]

q_hadamard = hadamard_transform(q, scale=1/math.sqrt(q.shape[-1]))
k_hadamard = hadamard_transform(k, scale=1/math.sqrt(k.shape[-1]))
v_hadamard = hadamard_transform(v, scale=1/math.sqrt(v.shape[-1]))
tensor_path = tensor_path.replace(".pt", "_hadamard.pt")
torch.save({
    "query": q_hadamard,
    "key": k_hadamard,
    "value": v_hadamard
}, tensor_path)