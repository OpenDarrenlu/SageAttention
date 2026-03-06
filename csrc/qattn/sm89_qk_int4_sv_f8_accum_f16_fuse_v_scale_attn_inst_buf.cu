#include "attn_cuda_sm89.h"
#include "qk_int_sv_f8_cuda_sm89.cuh"
#include <mutex>
#include <type_traits>

torch::Tensor qk_int4_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    torch::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    int kernel_config,
                    float sm_scale,
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(value_scale);

  CHECK_LASTDIM_CONTIGUOUS(query);
  CHECK_LASTDIM_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value); // ensure value is contiguous to prevent troubles in the kernel
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(value_scale);

  CHECK_DTYPE(query, torch::kInt8);
  CHECK_DTYPE(key, torch::kInt8);
  CHECK_DTYPE(query_scale, torch::kFloat32);
  CHECK_DTYPE(key_scale, torch::kFloat32);
  CHECK_DTYPE(value_scale, torch::kFloat32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value_scale, 3);

  const int batch_size = query.size(0);
  const int head_dim = output.size(3);
  const int packed_head_dim = head_dim / 2;

  if (head_dim % 2 != 0) {
    std::ostringstream err_msg;
    err_msg << "Output head dim must be even for int4 path, got " << head_dim;
    throw std::invalid_argument(err_msg.str());
  }

  int stride_bz_q = query.stride(0);
  int stride_bz_k = key.stride(0);
  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_h_q, stride_seq_k, stride_h_k, stride_h_v, stride_d_v, stride_seq_o, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);

    stride_seq_q = query.stride(1);
    stride_h_q = query.stride(2);
    stride_seq_k = key.stride(1);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_d_v = value.stride(1);
    stride_seq_o = output.stride(1);
    stride_h_o = output.stride(2);

    CHECK_SHAPE(query, batch_size, qo_len, num_qo_heads, packed_head_dim);
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, packed_head_dim);
    CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);
    assert(value.size(1) == head_dim);
    assert(value.size(2) == num_kv_heads);
  }
  else
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);

    stride_seq_q = query.stride(2);
    stride_h_q = query.stride(1);
    stride_seq_k = key.stride(2);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_d_v = value.stride(2);
    stride_seq_o = output.stride(2);
    stride_h_o = output.stride(1);

    CHECK_SHAPE(query, batch_size, num_qo_heads, qo_len, packed_head_dim);
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, packed_head_dim);
    CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
    assert(value.size(2) == head_dim);
    assert(value.size(1) == num_kv_heads);
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());
  }

  torch::Tensor lse = torch::empty({0});
  if (return_lse)
  {
    lse = torch::empty({batch_size, num_qo_heads, qo_len}, query.options().dtype(torch::kFloat32));
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;
  auto output_dtype = output.scalar_type();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;
            CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);

            auto launch_kernel = [&](auto cta_q, auto cta_k, auto warp_q, auto warp_k) {
              constexpr int CTA_Q = decltype(cta_q)::value;
              constexpr int CTA_K = decltype(cta_k)::value;
              constexpr int WARP_Q = decltype(warp_q)::value;
              constexpr int WARP_K = decltype(warp_k)::value;

              assert(value.size(0) == batch_size);
              assert(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K);

              if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
              {
                CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
                CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
              }
              else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
              {
                CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
                CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
              }
              else
              {
                static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity");
              }

              size_t smem_max = std::max(
                CTA_Q * (HEAD_DIM / 2) * sizeof(int8_t) + CTA_K * (HEAD_DIM / 2) * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t),
                CTA_Q * HEAD_DIM * sizeof(half));

              auto kernel_func = qk_int_sv_f8_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt4, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                          float, true, DTypeOut, ComputeUnit::kCudaCore, mask_mode, RETURN_LSE, true, false, true>;

              static std::once_flag attr_once;
              std::call_once(attr_once, [&]() {
                cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
              });

              dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
              dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

              kernel_func<<<grid, block, smem_max>>>(
                query.data_ptr<int8_t>(),
                key.data_ptr<int8_t>(),
                reinterpret_cast<int8_t*>(value.data_ptr()),
                reinterpret_cast<DTypeOut*>(output.data_ptr()),
                (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                reinterpret_cast<float*>(query_scale.data_ptr()),
                reinterpret_cast<float*>(key_scale.data_ptr()),
                reinterpret_cast<float*>(value_scale.data_ptr()),
                nullptr,
                qo_len,
                kv_len,
                num_kv_groups,
                stride_bz_q, stride_seq_q, stride_h_q,
                stride_bz_k, stride_seq_k, stride_h_k,
                stride_bz_v, stride_h_v, stride_d_v,
                stride_bz_o, stride_seq_o, stride_h_o,
                sm_scale);
            };

            switch (kernel_config)
            {
              case 0:
                launch_kernel(std::integral_constant<int, 128>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 32>{}, std::integral_constant<int, 64>{});
                break;
              case 1:
                launch_kernel(std::integral_constant<int, 128>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{});
                break;
              case 2:
                launch_kernel(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{});
                break;
              default:
              {
                std::ostringstream err_msg;
                err_msg << "Unsupported int4 kernel config " << kernel_config;
                throw std::invalid_argument(err_msg.str());
              }
            }
          });
        });
      });
    });
  });

  return lse;
}
