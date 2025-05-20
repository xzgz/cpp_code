// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <algorithm>
#include "hip_compat.h"
#include <iostream>

#define AT_DISPATCH_FP8_CASE(enum_type, ...) AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, fp8_t, __VA_ARGS__)

#define AITER_DISPATCH_CASE_FP8_TYPES(...)                         \
  AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__) \
  AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e4m3fnuz, __VA_ARGS__)

#define AITER_DISPATCH_FP8_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AITER_DISPATCH_CASE_FP8_TYPES(__VA_ARGS__))

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx942__))
  #define __HIP__MI300_MI250__
#endif

#if defined(__HIPCC__) && defined(__gfx942__)
  #define __HIP__MI300__
#endif

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

template <typename T>
struct scalar {};
template <>
struct scalar<c10::Half> {
  using type = half;
};
template <>
struct scalar<c10::BFloat16> {
  using type = __hip_bfloat16;
};

template <typename T>
struct scalar2 {};
template <>
struct scalar2<c10::Half> {
  using type = __half2;
};
template <>
struct scalar2<c10::BFloat16> {
  using type = __hip_bfloat162;
};

template <typename T>
struct fmul2_out {};
template <>
struct fmul2_out<c10::Half> {
  using type = __half2;
};
template <>
struct fmul2_out<c10::BFloat16> {
  using type = float2;
  // using type = __hip_bfloat162;
};

static bool is_fp8_ocp() {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  std::string device_arch = dprops->gcnArchName;
  size_t substring = device_arch.find("gfx94");
  return substring == std::string::npos;
}

template <typename T>
__device__ __forceinline__ float2 __s22float2(T v);

template <typename T>
__device__ __forceinline__ T __float2s(float v);

template <typename T>
__device__ __forceinline__ T __float22s2_rn(float2 v);

template <>
__device__ __forceinline__ half __float2s(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ float2 __s22float2(__half2 v) {
  return __half22float2(v);
}

template <>
__device__ __forceinline__ __half2 __float22s2_rn(float2 v) {
  return __float22half2_rn(v);
}

template <>
__device__ __forceinline__ __hip_bfloat16 __float2s(float v) {
  return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ float2 __s22float2(__hip_bfloat162 v) {
  return __bfloat1622float2(v);
}
template <>
__device__ __forceinline__ float2 __s22float2(float2 v) {
  return v;
}

template <>
__device__ __forceinline__ __hip_bfloat162 __float22s2_rn(float2 v) {
  return __float22bfloat162_rn(v);
}

__device__ __forceinline__ float __hmul_fp32(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) * __bfloat162float(b);
}
__device__ __forceinline__ float2 __hmul2_fp32(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return float2(__hmul_fp32(a.x, b.x), __hmul_fp32(a.y, b.y));
}
// __device__ __forceinline__ __hip_bfloat162 __hmul2_fp32(const __hip_bfloat162
// a, const __hip_bfloat162 b) {
//   return __hmul2(a, b);
// }
__device__ __forceinline__ __half2 __hmul2_fp32(const __half2 a, const __half2 b) { return __hmul2(a, b); }

__device__ __forceinline__ float __hfma_fp32(const __hip_bfloat16 a, const __hip_bfloat16 b, const float c) {
  return __ocml_fma_f32(__bfloat162float(a), __bfloat162float(b), c);
}
__device__ __forceinline__ float2 __hfma2_fp32(const __hip_bfloat162 a, const __hip_bfloat162 b, const float2 c) {
  return float2(__hfma_fp32(a.x, b.x, c.x), __hfma_fp32(a.y, b.y, c.y));
}
__device__ __forceinline__ __hip_bfloat162 __hfma2_fp32(const __hip_bfloat162 a, const __hip_bfloat162 b,
                                                        const __hip_bfloat162 c) {
  return __hfma2(a, b, c);
}
__device__ __forceinline__ __half2 __hfma2_fp32(const __half2 a, const __half2 b, const __half2 c) {
  return __hfma2(a, b, c);
}

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ float4 load_ntmprl(const float4* addr) {
  auto addr_alias = reinterpret_cast<const float*>(addr);
  auto dat0 = loadnt(addr_alias);
  auto dat1 = loadnt(addr_alias + 1);
  auto dat2 = loadnt(addr_alias + 2);
  auto dat3 = loadnt(addr_alias + 3);
  return make_float4(dat0, dat1, dat2, dat3);
}

// TBlock fetches entire rows of A, and entire col of B (K dimension); assume
// N=1 for time being grid is M/A_NUM_ROWS blocks
template <typename scalar_t, int NUM_A_ROWS_PER_BLOCK>
__global__ void LLGemm1_kernel(const scalar_t* in_a, const scalar_t* in_b, scalar_t* out_c, const int K) {
  using scalar2_t = typename scalar2<scalar_t>::type;
  using fmul2_out_t = typename fmul2_out<scalar_t>::type;

  auto af4 = reinterpret_cast<const float4*>(in_a);
  auto bf4 = reinterpret_cast<const scalar2_t*>(in_b);
  auto c = reinterpret_cast<scalar2_t*>(out_c);
  __shared__ float red_smem[NUM_A_ROWS_PER_BLOCK][WARP_SIZE];
  const int row_addr = blockIdx.x * NUM_A_ROWS_PER_BLOCK * K / 8;
  const int threadid = threadIdx.x;
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;
  const int qwarpid = threadid / num_warps;
  const int qthreadid = threadid % num_warps;
  float4 rowA_elem4[NUM_A_ROWS_PER_BLOCK];
  scalar2_t colB_elem4x, colB_elem4y, colB_elem4z, colB_elem4w;
  float acc[NUM_A_ROWS_PER_BLOCK];
  fmul2_out_t acch2;
  scalar2_t oval;

  // As we later use warp shuffle operations, we may have more threads in the
  // block than the actual available data, hence the if guard here.
  if (threadid * 8 < K) {
#pragma unroll
    for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
      // rowA_elem4[i] holds 8 * half numbers seen as a single float4.
      rowA_elem4[i] = load_ntmprl(&af4[row_addr + threadid + K / 8 * i]);
    }
  }

  colB_elem4x = bf4[threadid * 4 + 0];
  colB_elem4y = bf4[threadid * 4 + 1];
  colB_elem4z = bf4[threadid * 4 + 2];
  colB_elem4w = bf4[threadid * 4 + 3];

  scalar2_t Af2;
  [[maybe_unused]] scalar2_t Bf2;
  float2 S;

  auto Ah2ptr = reinterpret_cast<scalar2_t*>(&rowA_elem4);
  scalar2_t* ah2lptr;

#pragma unroll
  for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
    // Multiply-add on 8 scalar_t.
    ah2lptr = Ah2ptr + i * 4;
    Af2 = *(ah2lptr);
    acch2 = __hmul2_fp32(Af2, colB_elem4x);
    Af2 = *(ah2lptr + 1);
    acch2 = __hfma2_fp32(Af2, colB_elem4y, acch2);
    Af2 = *(ah2lptr + 2);
    acch2 = __hfma2_fp32(Af2, colB_elem4z, acch2);
    Af2 = *(ah2lptr + 3);
    acch2 = __hfma2_fp32(Af2, colB_elem4w, acch2);
    S = __s22float2(acch2);

    // See comment above concerning the if guard.
    acc[i] = (threadid * 8 < K ? S.x + S.y : 0.f);
  }

// all reduce across warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
#pragma unroll
    for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
      acc[i] += __shfl_xor(acc[i], mask);
    }
  }

  // Warp leaders store the data to shared memory.
  if (lane < NUM_A_ROWS_PER_BLOCK) {
    red_smem[lane][warp] = acc[lane];
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  if (qwarpid < NUM_A_ROWS_PER_BLOCK) {
    acc[qwarpid] = qthreadid < num_warps ? red_smem[qwarpid][qthreadid] : 0.f;
    for (int mask = num_warps / 2; mask >= 1; mask /= 2) {
      acc[qwarpid] += __shfl_xor(acc[qwarpid], mask);
    }
    float oval2 = __shfl_xor(acc[qwarpid], num_warps);

    if (lane % (num_warps * 2) == 0) {
      oval = __float22s2_rn<scalar2_t>(make_float2(acc[qwarpid], oval2));
      c[blockIdx.x * NUM_A_ROWS_PER_BLOCK / 2 + qwarpid / 2] = oval;
    }
  }
}

// template <typename T>
void LLGemm1(void* in_a, void* in_b, void* out_c, const int M, const int K, cudaStream_t stream,
             const int rows_per_block, const c10::ScalarType scalar_type) {
  // NUM_TREADS need to be a multiple of WARP_SIZE, as we are using warp shuffle
  // operations.
  const int NUM_THREADS = K * 2 / 16 % WARP_SIZE == 0 ? K * 2 / 16 : K * 2 / 16 + (WARP_SIZE - K * 2 / 16 % WARP_SIZE);

  int NUM_BLOCKS = M / rows_per_block;

  // call the kernel function...
  AT_DISPATCH_REDUCED_FLOATING_TYPES(scalar_type, "LLGemm1", [&] {
    scalar_t* a_ptr = reinterpret_cast<scalar_t*>(in_a);
    scalar_t* b_ptr = reinterpret_cast<scalar_t*>(in_b);
    scalar_t* c_ptr = reinterpret_cast<scalar_t*>(out_c);
    if (rows_per_block == 2) {
      LLGemm1_kernel<scalar_t, 2><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else if (rows_per_block == 4) {
      LLGemm1_kernel<scalar_t, 4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else if (rows_per_block == 8) {
      LLGemm1_kernel<scalar_t, 8><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else if (rows_per_block == 16) {
      LLGemm1_kernel<scalar_t, 16><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else {
      NUM_BLOCKS = M / 4;
      LLGemm1_kernel<scalar_t, 4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    }
  });
}
