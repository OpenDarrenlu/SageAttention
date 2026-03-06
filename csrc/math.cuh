/*
 * Copyright (c) 2024 by SageAttention team.
 * 
 * This file is based on code from Flashinfer, https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/math.cuh
 * Copyright (c) 2023 by FlashInfer team.
 * Small modifications made by SageAttention team, 2024 (e.g., renamed namespace).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef USHORT_TYPE
#define USHORT_TYPE
typedef unsigned short ushort;
#endif

namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;
constexpr float log2e_recp = 1.0f / log2e;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

constexpr int kSoftmaxE4M3Lut64Entries = 64;
constexpr float kSoftmaxE4M3Lut64MaxDelta = 8.807f;
constexpr float kSoftmaxE4M3Lut64Scale = (kSoftmaxE4M3Lut64Entries - 1) / kSoftmaxE4M3Lut64MaxDelta;

static __device__ __constant__ unsigned char kSoftmaxE4M3Lut64[kSoftmaxE4M3Lut64Entries] = {
    126, 125, 124, 122, 121, 121, 120, 118,
    117, 116, 115, 114, 113, 112, 110, 109,
    108, 107, 106, 105, 104, 103, 101, 100,
    99, 98, 97, 96, 95, 93, 92, 91,
    90, 89, 88, 87, 86, 84, 83, 82,
    81, 80, 79, 78, 77, 75, 74, 73,
    73, 72, 70, 69, 68, 67, 66, 65,
    64, 62, 61, 60, 59, 58, 57, 56,
};

__forceinline__ __device__ void load_softmax_e4m3_lut64_to_shared(unsigned char *smem_lut) {
  const uint32_t thread_linear_idx = threadIdx.x + blockDim.x * threadIdx.y;
  if (thread_linear_idx < kSoftmaxE4M3Lut64Entries) {
    smem_lut[thread_linear_idx] = kSoftmaxE4M3Lut64[thread_linear_idx];
  }
}

__forceinline__ __device__ int softmax_e4m3_lut64_index(float delta) {
  if (delta >= kSoftmaxE4M3Lut64MaxDelta) {
    return -1;
  }

  float clamped_delta = fmaxf(delta, 0.0f);
  int index = __float2int_rn(clamped_delta * kSoftmaxE4M3Lut64Scale);
  return max(0, min(kSoftmaxE4M3Lut64Entries - 1, index));
}

__forceinline__ __device__ unsigned char lookup_softmax_e4m3_lut64(float delta, const unsigned char *lut) {
  const int index = softmax_e4m3_lut64_index(delta);
  return (index < 0) ? 0 : lut[index];
}

__forceinline__ __device__ uint32_t pack_u8x4(
    unsigned char v0, unsigned char v1, unsigned char v2, unsigned char v3) {
  return static_cast<uint32_t>(v0) |
         (static_cast<uint32_t>(v1) << 8) |
         (static_cast<uint32_t>(v2) << 16) |
         (static_cast<uint32_t>(v3) << 24);
}

__forceinline__ __device__ uint32_t pack_softmax_e4m3_lut64(
    float delta0, float delta1, float delta2, float delta3, const unsigned char *lut) {
  return pack_u8x4(
      lookup_softmax_e4m3_lut64(delta0, lut),
      lookup_softmax_e4m3_lut64(delta1, lut),
      lookup_softmax_e4m3_lut64(delta2, lut),
      lookup_softmax_e4m3_lut64(delta3, lut));
}

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) {
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half ptx_exp2(half x) {
  ushort y_u16;
  asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly shuffle
 *   between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a butterfly
 *   shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ lane_mask]
 */
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  return __shfl_xor_sync(0xffffffff, x, lane_mask);
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes 1/sqrt(x)
 * \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ float tanh(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half2 tanh(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half tanh(half x) {
  ushort y_u16;
  asm volatile("tanh.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

}  // namespace math
