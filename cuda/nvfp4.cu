#include "nvfp4.hpp"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <algorithm>

namespace lb {

// FP4 E2M1 magnitude table (codes 0..7); sign is bit 3.
__constant__ float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float fp4_to_float(unsigned nib) {
  float m = kE2M1[nib & 7u];
  return (nib & 8u) ? -m : m;
}

__device__ __forceinline__ float fp8e4m3_to_float(uint8_t raw) {
  __nv_fp8_e4m3 v;
  v.__x = raw;
  return static_cast<float>(v);
}

// One thread per output element. Coalesced along the packed byte stream; the
// group scale is reused across 16 consecutive columns.
__global__ void dequant_nvfp4_kernel(const uint8_t* __restrict__ packed,
                                     const uint8_t* __restrict__ scale, float gscale,
                                     float* __restrict__ out, int64_t rows, int64_t in,
                                     int in2, int ggroups, int group) {
  int64_t total = rows * in;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    int64_t o = idx / in;
    int64_t i = idx - o * in;
    uint8_t byte = packed[o * in2 + (i >> 1)];
    unsigned nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    float sc = fp8e4m3_to_float(scale[o * ggroups + (int)(i / group)]);
    out[idx] = (fp4_to_float(nib) * sc) / gscale;  // op order matches the reference
  }
}

void dequant_nvfp4(const uint8_t* packed, const uint8_t* scale, float gscale,
                   float* out, int64_t rows, int64_t in, int group) {
  int in2 = (int)(in / 2);
  int gg = (int)(in / group);
  int block = 256;
  int64_t total = rows * in;
  int grid = (int)std::min<int64_t>(65535, (total + block - 1) / block);
  dequant_nvfp4_kernel<<<grid, block>>>(packed, scale, gscale, out, rows, in, in2, gg, group);
}

// y[m,o] = sum_i x[m,i] * dequant(W)[o,i]. One thread per output element; the
// weight row is dequanted on the fly, f32 accumulation. This is the correctness
// baseline — the FP4 tensor-core (PTX mma) kernel gets verified against it.
__global__ void gemm_nvfp4_kernel(const float* __restrict__ x, const uint8_t* __restrict__ packed,
                                  const uint8_t* __restrict__ scale, float gscale,
                                  float* __restrict__ y, int M, int OUT, int IN,
                                  int in2, int gg, int group) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;  // output neuron
  int m = blockIdx.y * blockDim.y + threadIdx.y;  // batch row
  if (o >= OUT || m >= M) return;
  const uint8_t* wrow = packed + (int64_t)o * in2;
  const uint8_t* srow = scale + (int64_t)o * gg;
  const float* xrow = x + (int64_t)m * IN;
  float acc = 0.f;
  for (int i = 0; i < IN; ++i) {
    uint8_t byte = wrow[i >> 1];
    unsigned nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    float w = (fp4_to_float(nib) * fp8e4m3_to_float(srow[i / group])) / gscale;
    acc += xrow[i] * w;
  }
  y[(int64_t)m * OUT + o] = acc;
}

void gemm_nvfp4(const float* x, const uint8_t* packed, const uint8_t* scale, float gscale,
                float* y, int M, int64_t out, int64_t in, int group) {
  int in2 = (int)(in / 2);
  int gg = (int)(in / group);
  dim3 block(64, 4);
  dim3 grid((unsigned)((out + block.x - 1) / block.x), (unsigned)((M + block.y - 1) / block.y));
  gemm_nvfp4_kernel<<<grid, block>>>(x, packed, scale, gscale, y, M, (int)out, (int)in,
                                     in2, gg, group);
}

}  // namespace lb
