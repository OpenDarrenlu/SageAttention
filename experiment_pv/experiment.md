接下来你需要做一个大大的实验，实验安排如下：

# 前置准备

1. 阅读 @SageAttention/sageattention/core_lut.py 所涉及的所有代码，理解lut_sageattn的原理（提示：QK仍采用int8，主要是P用较高精度，V用较低精度，PV矩阵乘仍采用V反量化为高精度，然后计算）
2. 阅读 1中所用到的量化代码 @SageAttention/sageattention/triton/quant_per_channel.py 这是对V做per-channel量化的原理
3. 注意，P的量化需要在 @SageAttention/sageattention/triton/attn_qk_int8_lut_v_int8.py 的核心kernel内做，而V可以在 @SageAttention/sageattention/core_lut.py 进入核心的attn计算前做

# workflow

1. 基于 @SageAttention/sageattention/triton/attn_qk_int8_lut_v_int8.py 进行修改，支持P的在线量化成 mxfp8, mxfp4, nvfp4, mxint4, int8, int4, fp16, bf16，如果triton不支持，可以采用软件模拟，即模拟出量化精度损失即可，量化粒度统一采用per-token量化
2. 基于 @SageAttention/sageattention/triton/quant_per_channel.py 实现V的int8，int4，int2的per-channel量化
3. @SageAttention/sageattention/triton/attn_qk_int8_lut_v_int8.py 中PV实现矩阵乘时，仍然反量化为fp16（即原精度）进行计算（因为我只需要模拟精度损失，计算时完全可以进行反量化）
4. 上述PV量化精度可以通过constexpr这种模版常量进行控制，避免编译开销，从 @SageAttention/sageattention/core_lut.py 中的核心接口都要串起来
5. 在 @SageAttention/scripts/work4 中实现对以上精度排列组合的attention计算精度误差进行评估。基于 @SageAttention/scripts/work3/modify_wan3.py 进行修改，在 @SageAttention/scripts/work4 中放代码，对Wan整个模型（假模型，仅仅为了评估精度和跑通）进行精度实验，确保能在模型中跑通，且精度误差较小。
6. 如果上述出现精度问题，请记录并改进量化粒度（比如block scaling），虽然目前的硬件可能不支持block scaling，但你可以通过软件模拟的方式（即反量化）来实现（因为我只需要考虑量化精度损失，不考虑mma指令计算时的不溢出情况下可以忽略不计的误差）
7. 请将上述代码修改原理，实验设计，实验结果等写到 @SageAttention/scripts/work4的一个文档里
8. 尽可能优化实验设计，直到一个合理的结果。并对实验结果进行分析，分析P V不同精度下的模型精度，找到一个兼具精度和更低比特的量化算法（包含量化比特数，量化粒度，量化算法等）

# requierment
1. 实验要有理有据，做好实验设计和评估
2. 写kernel要注意实现的性能（可以参考 @~/workspace/agent-gpu-skills/ 下的skills） ，尽可能减少实验的时间成本
3. 对实验结果的分析也要有理有据
4. 我没空插足你的工作，以上实验很明确，请你一直运行，不必请求我，如果真遇到不明确的，你根据实验思想选择最优方案即可。