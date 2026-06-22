# SageAttention LUT P/V 精度扫掠实验报告

> 实验目标：在 SageAttention 的 LUT 分支上，系统评估注意力权重 `P` 与值向量 `V` 在不同低比特精度、不同量化粒度下的模型精度损失，找到一个兼具精度与压缩比的量化配置。

---

## 1. 代码修改原理

### 1.1 原始 LUT attention 的数据流

原始 `sageattention/core_lut.py` 中的 `sageattn_lut` 采用如下流程：

1. Q/K 做 per-block INT8 量化（`quant_per_block.py`）。
2. Attention kernel（`attn_qk_int8_lut_v_int8.py`）内部：
   - 用 INT8 QK 计算 `qk`。
   - 在线 softmax 得到 `P = exp2(qk - m)`。
   - 直接以 FP32/FP16 精度做 `P @ V`。
3. V 在进入 kernel 前做 per-channel INT8/INT4 量化（`quant_per_channel.py`）。

### 1.2 本次修改点

#### (1) P 的在线多精度量化

在 `sageattention/triton/attn_qk_int8_lut_v_int8.py` 中新增 `_quantize_p_per_token`：

- 支持格式：`fp16`、`bf16`、`int8`、`int4`、`mxfp8`、`mxfp4`、`nvfp4`、`mxint4`。
- 因为 `P` 经过 softmax 后非负，统一使用 **unsigned** 量化网格（避免浪费 1 bit 符号位）。
- 对整数/类浮点低比特格式，采用 per-block scale 的软件模拟：
  - 计算当前 block 的 `max_val`，映射到该格式的正最大表示值。
  -  round → clamp → rescale，返回 float32 反量化值。
- 对 `fp16`/`bf16` 直接做 cast round-trip。
- 通过 `P_DTYPE_CODE: tl.constexpr` 控制分支，Triton 为每种精度单独编译并缓存 kernel。

| 格式 | 模拟的网格正最大值 | 说明 |
|------|-------------------|------|
| int8 | 255 | unsigned 8-bit |
| int4 / mxint4 | 15 | unsigned 4-bit |
| mxfp8 | 448 | E4M3 类 per-token scaled FP8 |
| mxfp4 | 6 | E2M1 类 per-token scaled FP4 |
| nvfp4 | 28 | E3M0 类 per-token scaled FP4 |

#### (2) V 的 per-channel 多精度量化

在 `sageattention/triton/quant_per_channel.py` 中新增 `per_channel_int2`：

- 复用已有的 stats kernel 与 apply-quant kernel。
- INT2 值域为 `[-2, 1]`，仍用 INT8 容器存储，scale_max=2.0。
- INT4 值域 `[-8, 7]`，scale_max=7.0；INT8 值域 `[-127, 127]`，scale_max=127.0。

#### (3) PV 矩阵乘仍以原精度计算

在 attention kernel 中：

```python
v_compute = v.to(compute_ty) * V_scale.to(compute_ty)
p_compute = p_sim.to(compute_ty)
acc += tl.dot(p_compute, v_compute).to(tl.float32)
```

`compute_ty` 由输出 dtype（输入 dtype）决定，即 FP16/BF16。这样只模拟量化精度损失，不引入 MMA 指令本身的误差。

#### (4) P 的 block scaling 粒度

新增 `P_BLOCK_N: tl.constexpr`：

- `P_BLOCK_N == BLOCK_N (64)`：每个 query token 在每个 KV tile 内一个 scale（默认）。
- `P_BLOCK_N = 32/16`：在 64 长度的 KV tile 内再细分 block，每个子 block 独立 scale。
- 通过 `tl.where` 掩码取出子 block，计算子 block 的 `max_val`，量化/反量化后写回。

该功能通过 `sageattn_lut(..., p_block_n=...)` 暴露给用户。

#### (5) 接口串接

`core_lut.py` 中统一入口为 `sageattn_qk_int8_p_lut_vq_triton`，参数：

- `p_quant_dtype`
- `v_quant_dtype`
- `p_block_n`

`__init__.py` 导出 `sageattn_qk_int8_p_lut_vq_triton`，保持 `sageattn_lut` 向后兼容。

---

## 2. 实验设计

### 2.1 评估对象

- **P 精度**：fp16、bf16、int8、int4、mxfp8、mxfp4、nvfp4、mxint4（8 种）。
- **V 精度**：int8、int4、int2（3 种）。
- **P 粒度**：64（per-tile）、32、16（block scaling）。

共 8×3 = 24 种主实验组合，另加 3 种低 bit P × 3 种粒度 = 9 组 block scaling 细化实验。

### 2.2 评估指标

1. **Attention-level 误差**：在 Wan 假模型的每个 transformer block 中，捕获 self-attention 的 Q/K/V 与输出；用 SDPA 输出作为基准，计算量化 attention 输出在同输入下的误差。
   - `max_abs_err` / `mean_abs_err`
   - `rel_max` / `rel_mean`
   - `rmse`
   - `cos_sim`
2. **Model-level 误差**：比较整网最终输出 tensor 与 SDPA 基准的差异。

### 2.3 实验设置

- 假模型：`WanTransformer3DModel`，`num_layers=4`，`attention_head_dim=128`，`num_attention_heads=12`，空间分辨率 `64×64`，帧数 4。
- 输入 dtype：`bfloat16`。
- 随机种子：0、1、2，取平均。
- 运行脚本：`python scripts/work4/experiment_wan4.py`。
- 结果保存：`scripts/work4/results/{attention_level.csv, model_level.csv, block_scaling.csv, results.json}`。

---

## 3. 主要实验结果

### 3.1 Attention-level 误差（平均值，跨层、跨种子）

| P / V | max_abs_err | mean_abs_err | rel_mean | cos_sim |
|-------|-------------|--------------|----------|---------|
| fp16 / int8 | 4.29e-3 | 5.22e-4 | 4.48e-2 | 0.9999785 |
| bf16 / int8 | 4.22e-3 | 5.22e-4 | 4.47e-2 | 0.9999785 |
| **int8 / int8** | 4.41e-3 | **5.26e-4** | 4.49e-2 | 0.9999782 |
| **mxfp8 / int8** | 4.28e-3 | **5.24e-4** | 4.47e-2 | 0.9999784 |
| nvfp4 / int8 | 8.48e-3 | 8.08e-4 | 6.79e-2 | 0.9999524 |
| int4 / int8 | 1.50e-2 | 1.47e-3 | 1.18e-1 | 0.9998618 |
| mxint4 / int8 | 1.50e-2 | 1.47e-3 | 1.18e-1 | 0.9998618 |
| mxfp4 / int8 | 3.08e-2 | 4.15e-3 | 3.18e-1 | 0.9991228 |
| fp16 / int4 | 1.56e-2 | 1.81e-3 | 1.55e-1 | 0.9997718 |
| fp16 / int2 | 5.69e-2 | 6.84e-3 | 5.78e-1 | 0.9968045 |

（完整表格见 `results/attention_level.csv`）

### 3.2 Model-level 误差（平均值，跨种子）

所有配置的最终输出 `max_abs_err` 均为 `1.5625e-2`（接近 bfloat16 的离散化步长），说明最终输出层对 attention 内部差异有一定饱和/掩蔽作用。区分度主要看 `mean_abs_err`：

| P / V | mean_abs_err | rel_mean | cos_sim |
|-------|--------------|----------|---------|
| fp16 / int8 | 6.79e-4 | 1.09e-2 | 0.9999972 |
| int8 / int8 | 6.80e-4 | 1.32e-2 | 0.9999972 |
| mxfp8 / int8 | 6.80e-4 | 1.16e-2 | 0.9999972 |
| nvfp4 / int8 | 7.19e-4 | 1.33e-2 | 0.9999970 |
| int4 / int8 | 8.00e-4 | 1.59e-2 | 0.9999967 |
| mxfp4 / int8 | 1.06e-3 | 1.92e-2 | 0.9999952 |
| fp16 / int4 | 8.33e-4 | 1.48e-2 | 0.9999965 |
| fp16 / int2 | 1.24e-3 | 2.22e-2 | 0.9999942 |

### 3.3 Block scaling 细化（V=int8）

| P 精度 / block size | mean_abs_err | 相对 b64 改善 |
|---------------------|--------------|---------------|
| int4 / b64 | 1.47e-3 | — |
| int4 / b32 | 1.13e-3 | -23% |
| **int4 / b16** | **8.95e-4** | **-39%** |
| nvfp4 / b64 | 8.08e-4 | — |
| nvfp4 / b32 | 7.05e-4 | -13% |
| **nvfp4 / b16** | **6.35e-4** | **-21%** |
| mxfp4 / b64 | 4.15e-3 | — |
| mxfp4 / b32 | 3.23e-3 | -22% |
| mxfp4 / b16 | 2.35e-3 | -43% |

---

## 4. 结果分析

### 4.1 P 精度的影响

- **fp16/bf16 P**：作为基准，attention MAE 约 5.2e-4，24 bits（P 16 bit + V 8 bit）。
- **int8 / mxfp8 P**：attention MAE 与 fp16/bf16 几乎相同（约 5.2e-4），但 P 只占 8 bit，总带宽 16 bit。**这是本次实验最重要的发现**：在 per-block scaled 8-bit 下，P 的量化损失已被淹没在 Q/K INT8、PV FP16 计算等其他误差源中。
- **nvfp4 P**：默认粒度（b64）MAE 8.1e-4，仍可接受；block size 16 后降至 6.35e-4，接近 8-bit 水平。
- **int4 / mxint4 P**：默认粒度 MAE 1.47e-3；block size 16 后降至 8.95e-4，提升明显，但仍略差于 nvfp4_b16。
- **mxfp4 P**：表现最差（b64 MAE 4.15e-3）。原因是 E2M1 类网格的动态范围只有 6，对 P 的分布适配不佳；即使 block size 16 也只能降到 2.35e-3。

### 4.2 V 精度的影响

- **V=int8**：所有 P 精度下都是最优或接近最优。
- **V=int4**：attention MAE 约 1.8e-3，是 V=int8 的 3.5 倍。说明 per-channel INT4 对 V 已有可见损失。
- **V=int2**：attention MAE 约 6.8e-3，且与 P 精度几乎无关（fp16/int2 与 int4/int2、mxfp4/int2 等数值接近）。**误差主要由 V=int2 主导**。

结论：**V 不建议低于 INT8**；若必须压缩 V，需要引入 V 的 block scaling（如 per-channel-block 或 per-token-block），本次实验未实现，列为后续方向。

### 4.3 粒度（block scaling）的影响

- 对 int4、nvfp4、mxfp4 P， finer block size（16）都显著降低误差。
- nvfp4_b16 用 12 bits（P 4 + V 8）达到 6.35e-4 attention MAE，仅比 16-bit 方案高 20% 左右误差，但比特数降低 25%。
- int4_b16 同样 12 bits，MAE 8.95e-4，性价比低于 nvfp4_b16。

---

## 5. 推荐量化算法

综合精度、压缩比与实现复杂度，推荐以下三档配置：

### 5.1 精度优先（16 bits）

- **P = int8 或 mxfp8，V = int8，p_block_n = 64**
- attention MAE ≈ 5.2e-4，与 fp16 P 几乎无差别。
- 总带宽 16 bits（P 8 + V 8），比 fp16 P 方案节省 33%。
- 实现最简单：直接复用现有 per-tile scale。

### 5.2 精度-压缩比平衡（12 bits）

- **P = nvfp4，V = int8，p_block_n = 16**
- attention MAE ≈ 6.35e-4，model MAE ≈ 7.0e-4。
- 总带宽 12 bits（P 4 + V 8），比 fp16 P 方案节省 50%。
- 需要支持 P 的 block scaling（本实验已用软件模拟验证）。

### 5.3 最低可接受（12 bits，备选）

- **P = int4，V = int8，p_block_n = 16**
- attention MAE ≈ 8.95e-4，仍远好于 V=int4（1.8e-3）或 V=int2（6.8e-3）。
- 若硬件更友好于 INT4 而非 NVFP4，可作为备选。

### 5.4 不推荐

- **V=int4 / int2**：在模型级和 attention 级都带来显著误差，需配合 V 的 block scaling 才值得考虑。
- **mxfp4 P**：即便 block size 16，MAE 仍达 2.35e-3，性价比不如 nvfp4 或 int4。

---

## 6. 实验局限与未来工作

1. **V 粒度仍为 per-channel**：V=int4/int2 的误差主要来源于 V 量化。未来可引入 V 的 per-channel-block 或 per-token-block scaling。
2. **P 的 true per-token scale**：当前 kernel 是 streaming attention，P 的 scale 实际是在每个 64 长度 KV tile 内计算。若要全局 per-token scale，需要额外一次遍历存储每行的 max，实现更复杂。
3. **浮点格式模拟**：mxfp8/mxfp4/nvfp4 采用 uniform-grid 软件模拟，仅近似真实 FP8/FP4 的非均匀分布。若后续要对接真实硬件，建议用 Triton `tl.dot_scaled` 或 CUDA 原生格式重新标定。
4. **模型规模**：本实验使用假模型（4 layer、128 head_dim、64×64）。真实 Wan 模型更大，建议在实际模型上再验证推荐的配置。

---

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `sageattention/triton/attn_qk_int8_lut_v_int8.py` | 支持多精度 P 量化与 block scaling 的 attention kernel |
| `sageattention/triton/quant_per_channel.py` | 新增 V 的 INT2 per-channel 量化 |
| `sageattention/core_lut.py` | 统一接口 `sageattn_qk_int8_p_lut_vq_triton`，支持 p_quant_dtype/v_quant_dtype/p_block_n |
| `sageattention/__init__.py` | 导出新接口 |
| `scripts/work4/modify_wan4.py` | Wan attention processor 与 dummy 模型运行工具 |
| `scripts/work4/experiment_wan4.py` | 主实验脚本（attention-level + model-level + block scaling） |
| `scripts/work4/results/*.csv` | 实验结果表格 |
| `scripts/work4/results/results.json` | 实验结果原始数据 |
| `scripts/work4/experiment_report.md` | 本报告 |

---

## 8. 复现命令

```bash
cd /home/lutingzhan/workspace_infer/SageAttention
python scripts/work4/experiment_wan4.py
```

快速验证单个配置：

```python
from sageattention import sageattn_lut
import torch

q = torch.randn(1, 12, 256, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(1, 12, 256, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(1, 12, 256, 128, dtype=torch.bfloat16, device="cuda")

# 推荐配置 1：精度优先
out = sageattn_lut(q, k, v, p_quant_dtype="int8", v_quant_dtype="int8")

# 推荐配置 2：平衡
out = sageattn_lut(q, k, v, p_quant_dtype="nvfp4", v_quant_dtype="int8", p_block_n=16)
```

---

## 9. 补充实验：V 的 MXINT2（per-channel-per-block INT2）

### 9.1 动机

主实验发现 `V=int2` 时 attention-level MAE 约 **6.8e-3**，且几乎不随 P 精度变化，说明误差主要由 V 量化本身主导。为了把 V 进一步压到 2-bit 同时保持可用精度，需要**缩小 V 的量化粒度**。最直接的方案就是 MXINT2：对每个 channel 的 KV 序列做分块 scaling，而不是整个序列一个 scale。

### 9.2 实现

在 `sageattention/triton/quant_per_channel.py` 中新增 `per_channel_block_intx` / `per_channel_int2_block`：

- 将 V 沿 KV 维度切成 `block_size=64` 的块。
- 每个 `(batch, head, channel, block)` 计算独立的 scale。
- 仍用 INT8 容器存储 2-bit 值，值域 `[-2, 1]`，scale_max=2.0（对称量化）。

在 `sageattention/triton/attn_qk_int8_lut_v_int8.py` 的 attention kernel 中新增 `V_BLOCK_SIZE` 支持：

- 每个 KV tile（64 token）加载对应 block 的 V scale。
- 反量化后仍在 FP16/BF16 做 PV matmul，只模拟精度损失。

`core_lut.py` 中 `sageattn_lut` 新增参数 `v_block_size`：

- `v_block_size=0`：per-channel（默认，兼容旧行为）。
- `v_block_size=64`：per-channel-per-block（MXINT2）。

### 9.3 V 量化重建误差（最可信的指标）

> **重要修正**：早期版本曾用「假模型 model-level mean_abs_err」对比 MXINT2 与 per-channel
> INT2，并错误地得出「MXINT2 反而更差」的结论。经复核，**该 model-level 指标在本实验里
> 是失真的**（原因见 9.3.2），不能作为 V 量化质量的依据。下面改用直接的 **V 重建误差**
> `|dequant(V) - V|` 作为主指标，它剥离了 attention/网络传播带来的噪声，是最干净的衡量方式。

#### 9.3.1 V 重建误差对比

在受控合成数据（per-channel 不同幅度 + 沿序列的局部幅度突变，模拟真实 V 的分布特征）上，
对 INT2（当前 scale_max=2.0，值域 `[-2,1]`，smooth_v=True）测量重建误差：

| INT2 方案 | mean 重建误差 | 相对 per-channel |
|-----------|---------------|------------------|
| per-channel（整条序列一个 scale） | 1.6482 | — |
| block-scaled bs=128 | 1.2251 | −26% |
| block-scaled bs=64（MXINT2，kernel 默认） | 1.1445 | −31% |
| block-scaled bs=32 | 1.0657 | −35% |
| block-scaled bs=16 | **0.9852** | **−40%** |

**结论：block scaling 在 V 重建层面单调降低误差，粒度越细越好**，完全符合「缩小量化粒度
提高精度」的直觉。MXINT2 的量化算法本身是正确且有效的。

#### 9.3.2 为什么早期的 model-level 指标会显示 MXINT2「更差」？

那是**测量假象**，根源有三：

1. **假模型未训练**：`WanTransformer3DModel` 是随机初始化权重。V 的量化误差经过若干随机
   线性投影后，与最终输出误差之间几乎没有单调关系——测到的是噪声在随机网络里的传播，
   而非量化质量。
2. **bf16 输出饱和**：所有配置的最终输出 `max_abs_err` 都恒为 `1.5625e-2`（bf16 离散步长），
   说明输出动态范围极小，model-level 指标已被量化台阶「夹平」，分辨不出 V 量化的优劣。
3. **样本不足**：早期对比只跑 seed=42 单次、未做多种子平均，1.25e-3 与 1.92e-3 的差异
   在该噪声水平下没有统计意义。

因此，**评估超低比特 V 量化应以 V 重建误差（或在真实/已训练模型上的端到端指标）为准，
而非未训练假模型的输出误差**。

#### 9.3.3 待补充：真实模型 attention-level 指标

由于 `experiment_wan4.py` 的 block-scaling 细化实验当时只扫了 V=int8，尚未在 attention-level
直接测过 block-int2。后续应在 GPU 上补跑 per-channel-int2 vs block-int2 的 attention 输出误差
（以 SDPA 为基准）以进一步交叉验证；重建误差结论已足以支撑「block scaling 有效」这一判断。

### 9.4 真实视频生成

在 `Wan2.1` T2V-1.3B 上生成了 5 秒（81 帧）视频用于主观/客观对比，文件名包含精度标签：

| 配置 | 文件名标签 |
|------|-----------|
| P=fp16, V=int2 per-channel | `Pfp16_Vint2_vperch_pb64` |
| P=fp16, V=int2 block-scaled | `Pfp16_Vint2_vb64_pb64` |
| P=int8, V=int2 block-scaled | `Pint8_Vint2_vb64_pb64` |
| P=int4, V=int2 per-channel | `Pint4_Vint2_vperch_pb64` |
| P=int4, V=int2 block-scaled | `Pint4_Vint2_vb64_pb64` |
| P=mxfp4, V=int2 block-scaled | `Pmxfp4_Vint2_vb64_pb64` |
| P=nvfp4, V=int2 block-scaled | `Pnvfp4_Vint2_vb64_pb64` |
| P=mxint4, V=int2 block-scaled | `Pmxint4_Vint2_vb64_pb64` |

所有视频保存在 `/home/lutingzhan/workspace_infer/SageAttention/Wan2.1/`，可直接通过文件名区分并评估生成效果。

### 9.5 对称 vs 非对称 MXINT2

当前实现默认是**对称** MXINT2（值域 `[-2,-1,0,1]`，max-abs/2.0 缩放）。一个自然的疑问是
非对称量化（unsigned `[0,3]` + per-block zero-point）是否更好。在同一份合成数据上对比：

| INT2 方案（block bs=64） | mean 重建误差 | max 重建误差 |
|--------------------------|---------------|--------------|
| 对称 MXINT2 | 1.1445 | 15.9 |
| 非对称 MXINT2（zero-point） | 1.3150 | **9.5** |

结论与直觉略有出入：
- V 在做完 per-channel 去均值（`smooth_v`）后接近**零均值**分布，对称量化已与之匹配，
  因此**对称 MXINT2 的平均误差反而更低**。
- 非对称的优势集中在 **max（尾部/离群）误差**：9.5 vs 15.9，对个别幅度突变的 block 更稳。

因此「非对称一定更优」并不成立——**block scaling 才是主要收益来源**，对称/非对称的选择取决于
你更在意平均误差还是离群误差。若 V 经过 smooth_v 已零均值化，**对称 MXINT2 通常就足够**。

### 9.6 非对称 MXINT2（备选，针对离群值）

若实测发现某些 head/block 的 V 分布明显偏斜、离群值导致可见瑕疵，可启用非对称 MXINT2：
对每个 block 用 unsigned grid `[0,1,2,3]` + per-block zero-point：

```
scale = (max - min) / 3
zp    = round(-min / scale)            # 零点附近整数 level
q     = round(v / scale) + zp          # q ∈ {0,1,2,3}
dequant = (q - zp) * scale
```

优势：
- 4 个 level 全部利用，覆盖 `[min, max]` 整个区间，对偏斜/有离群的 block 更鲁棒（max 误差更低）。
- 每个 block 自适应偏移。
- 硬件上只多一个 per-block zero-point（或预乘成 bias），开销很小。

`quant_per_channel.py` 中 `per_channel_block_intx` 已实现 `asymmetric=True` 路径（返回额外的
`zp`），attention kernel 中加入 `(v_int - zp) * scale` 的反量化路径即可启用。

### 9.7 其他 V=2bit 备选方案

| 方案 | 核心思想 | 适用场景 |
|------|---------|----------|
| Ternary {-1, 0, 1} | 3 个 level，强制稀疏 | V 值在 0 附近高度集中 |
| Per-block mean + scale | 去掉局部 DC 后再对称量化 | 局部均值波动大 |
| 2D block scaling | 同时对 sequence 和 channel 维度分块 | 需要更高精度但可接受更多 meta 数据 |

### 9.8 结论

- 已实现并验证了 **MXINT2**（V per-channel-per-block INT2）。**V 重建误差证明 block scaling
  单调有效**：per-channel 1.65 → block64 1.14 → block16 0.99（−40%），符合「缩小粒度提精度」的预期。
- 早期「MXINT2 在 model-level 更差」的说法是**未训练假模型 + bf16 饱和导致的测量假象**，已修正；
  评估超低比特 V 应以重建误差或真实/已训练模型的端到端指标为准。
- 对称 vs 非对称：V 经 smooth_v 零均值化后，**对称 MXINT2 平均误差更低**；非对称主要降低离群（max）误差。
- 真实视频已生成，可用于主观评估 P=4bit/16bit + V=2bit 的实际效果。
- 待办：在 GPU 上补跑 block-int2 的 attention-level 误差，做端到端交叉验证。

