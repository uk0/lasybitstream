#include "nvfp4.hpp"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <cstdlib>

namespace lb {

// FP4 E2M1 magnitude table (codes 0..7); sign is bit 3.
__constant__ float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float fp4_to_float(unsigned nib) {
  float m = kE2M1[nib & 7u];
  return (nib & 8u) ? -m : m;
}

// Hardware FP4 (e2m1) x2 -> f32 x2 decode (Blackwell `cvt.rn.f16x2.e2m1x2`): the
// low nibble decodes to .x, the high nibble to .y — matching the pack convention
// (low nibble = even column). Removes the scalar LUT ALU from the GEMV hot loop.
__device__ __forceinline__ float2 e2m1x2_to_f2(uint8_t byte) {
  __nv_fp4x2_e2m1 v; v.__x = byte;
  return __half22float2(static_cast<__half2>(v));
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

__device__ __forceinline__ uint8_t get_byte(uint4 p, int b) {
  uint32_t w = (b < 8) ? ((b < 4) ? p.x : p.y) : ((b < 12) ? p.z : p.w);
  return (uint8_t)(w >> (8 * (b & 3)));
}

// Bandwidth-optimal M=1 (decode) GEMV: one warp per output row, K tiles of 1024.
// Each lane streams a contiguous 32-column chunk as one 128-bit weight load
// (32 fp4 = 16 bytes), applies the two group-16 fp8 scales, and fmaf-accumulates
// against the shared activation tile; a warp shuffle reduces the row. (researcher plan)
template <int WARPS>
__global__ void gemv_nvfp4_m1_kernel(const float* __restrict__ x, const uint8_t* __restrict__ packed,
                                     const uint8_t* __restrict__ scale, float gscale,
                                     float* __restrict__ y, int OUT, int IN, int in2, int gg) {
  __shared__ float xs[32][33];                 // 1024 activations; pad avoids bank conflicts
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int o = blockIdx.x * WARPS + warp;
  const uint8_t* wr = (o < OUT) ? packed + (int64_t)o * in2 : nullptr;
  const uint8_t* sr = (o < OUT) ? scale + (int64_t)o * gg : nullptr;
  float acc = 0.f;
  for (int k0 = 0; k0 < IN; k0 += 1024) {
    int q = threadIdx.x * 4;                    // 256 threads * 4 = 1024 activations
    float4 a = *reinterpret_cast<const float4*>(x + k0 + q);
    int owner = q >> 5, j = q & 31;
    xs[owner][j] = a.x; xs[owner][j + 1] = a.y; xs[owner][j + 2] = a.z; xs[owner][j + 3] = a.w;
    __syncthreads();
    if (o < OUT) {
      int col = k0 + lane * 32;
      uint4 p = *reinterpret_cast<const uint4*>(wr + (col >> 1));
      uint16_t raw2 = *reinterpret_cast<const uint16_t*>(sr + (col >> 4));
      float s0 = fp8e4m3_to_float((uint8_t)(raw2 & 0xff)) / gscale;
      float s1 = fp8e4m3_to_float((uint8_t)(raw2 >> 8)) / gscale;
      #pragma unroll
      for (int b = 0; b < 16; ++b) {
        float2 w2 = e2m1x2_to_f2(get_byte(p, b));
        float s = (b < 8) ? s0 : s1;
        acc = fmaf(xs[lane][2 * b], w2.x * s, acc);
        acc = fmaf(xs[lane][2 * b + 1], w2.y * s, acc);
      }
    }
    __syncthreads();
  }
  for (int d = 16; d; d >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, d);
  if (lane == 0 && o < OUT) y[o] = acc;
}

// Weight-stationary batched GEMM (2..MAXB rows): one warp per output row loads
// each packed weight ONCE (uint4), unpacks to registers, and dots it against all
// M activation vectors (L2-resident). The weight read (~22 GB/token) is amortized
// across the whole batch, so aggregate throughput scales with M — this is the core
// of aggregate batching (500+ tok/s/card) as well as MTP verify / short prefills.
static const int MAXB = 32;
template <int WARPS>
__global__ void gemm_nvfp4_smallM_kernel(const float* __restrict__ x, const uint8_t* __restrict__ packed,
                                         const uint8_t* __restrict__ scale, float gscale,
                                         float* __restrict__ y, int M, int OUT, int IN, int in2, int gg) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int o = blockIdx.x * WARPS + warp;
  if (o >= OUT) return;
  const uint8_t* wr = packed + (int64_t)o * in2;
  const uint8_t* sr = scale + (int64_t)o * gg;
  float acc[MAXB]; for (int m = 0; m < M; ++m) acc[m] = 0.f;
  for (int k0 = 0; k0 < IN; k0 += 1024) {
    int col = k0 + lane * 32;
    uint4 p = *reinterpret_cast<const uint4*>(wr + (col >> 1));
    uint16_t raw2 = *reinterpret_cast<const uint16_t*>(sr + (col >> 4));
    float s0 = fp8e4m3_to_float((uint8_t)(raw2 & 0xff)) / gscale;
    float s1 = fp8e4m3_to_float((uint8_t)(raw2 >> 8)) / gscale;
    float wv[32];
    #pragma unroll
    for (int b = 0; b < 16; ++b) {
      float2 w2 = e2m1x2_to_f2(get_byte(p, b)); float s = (b < 8) ? s0 : s1;
      wv[2 * b] = w2.x * s; wv[2 * b + 1] = w2.y * s;
    }
    for (int m = 0; m < M; ++m) {
      const float* xr = x + (int64_t)m * IN + col;
      float a = 0.f;
      #pragma unroll
      for (int c = 0; c < 32; ++c) a += xr[c] * wv[c];
      acc[m] += a;
    }
  }
  for (int m = 0; m < M; ++m) {
    float v = acc[m];
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    if (lane == 0) y[(int64_t)m * OUT + o] = v;
  }
}

// Shared-staged batched GEMM (2..STAGEM rows): the whole block cooperatively
// stages the M activation vectors of each K-tile into shared ONCE (padded [32][33]
// to avoid bank conflicts), then every warp loads its weight tile ONCE (uint4) and
// dots it against all M staged rows. The weight read (~22 GB/token, the bottleneck)
// is amortized across the batch — aggregate tok/s scales with M up to the compute
// crossover. This is the core batched-decode kernel for aggregate throughput.
static const int STAGEM = 16;                          // <= 67 KB shared (opt-in)
template <int WARPS>
__global__ void gemm_nvfp4_stage_kernel(const float* __restrict__ x, const uint8_t* __restrict__ packed,
                                        const uint8_t* __restrict__ scale, float gscale,
                                        float* __restrict__ y, int M, int OUT, int IN, int in2, int gg) {
  extern __shared__ float xs[];                        // [M][32][33]
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int o = blockIdx.x * WARPS + warp;
  const uint8_t* wr = (o < OUT) ? packed + (int64_t)o * in2 : nullptr;
  const uint8_t* sr = (o < OUT) ? scale + (int64_t)o * gg : nullptr;
  float acc[STAGEM]; for (int m = 0; m < M; ++m) acc[m] = 0.f;
  for (int k0 = 0; k0 < IN; k0 += 1024) {
    for (int idx = threadIdx.x; idx < M * 1024; idx += blockDim.x) {   // stage M activation rows
      int m = idx >> 10, c = idx & 1023;
      xs[m * 32 * 33 + (c >> 5) * 33 + (c & 31)] = x[(int64_t)m * IN + k0 + c];
    }
    __syncthreads();
    if (o < OUT) {
      int col = k0 + lane * 32;
      uint4 p = *reinterpret_cast<const uint4*>(wr + (col >> 1));
      uint16_t raw2 = *reinterpret_cast<const uint16_t*>(sr + (col >> 4));
      float s0 = fp8e4m3_to_float((uint8_t)(raw2 & 0xff)) / gscale;
      float s1 = fp8e4m3_to_float((uint8_t)(raw2 >> 8)) / gscale;
      float wv[32];
      #pragma unroll
      for (int b = 0; b < 16; ++b) { float2 w2 = e2m1x2_to_f2(get_byte(p, b)); float s = (b < 8) ? s0 : s1;
        wv[2 * b] = w2.x * s; wv[2 * b + 1] = w2.y * s; }
      for (int m = 0; m < M; ++m) {
        const float* xr = xs + m * 32 * 33 + lane * 33;
        float a = 0.f;
        #pragma unroll
        for (int c = 0; c < 32; ++c) a += xr[c] * wv[c];
        acc[m] += a;
      }
    }
    __syncthreads();
  }
  for (int m = 0; m < M; ++m) { float v = acc[m];
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    if (lane == 0 && o < OUT) y[(int64_t)m * OUT + o] = v; }
}

// Tensor-core (bf16 WMMA) batched GEMM. Each warp computes a 16(M)x16(N) output
// tile: per K-16 step it dequants the FP4 weight tile -> bf16 in shared and casts
// the activation tile -> bf16, then mma_sync. The bf16 tensor cores raise the
// compute ceiling ~20x over f32 FMA, so batching becomes weight-bandwidth-bound
// again (M=64 -> ~800 tok/s). W4A16 with bf16 activations.
namespace wmma = nvcuda::wmma;
__global__ void gemm_nvfp4_wmma_kernel(const float* __restrict__ x, const uint8_t* __restrict__ packed,
                                       const uint8_t* __restrict__ scale, float gscale,
                                       float* __restrict__ y, int M, int OUT, int IN, int in2, int gg) {
  int nt = blockIdx.x * 16, mt = blockIdx.y * 16;      // output-neuron / batch-row tile origins
  __shared__ __nv_bfloat16 As[16 * 16], Bs[16 * 16];
  __shared__ float Cs[16 * 16];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
  wmma::fill_fragment(cf, 0.f);
  for (int k0 = 0; k0 < IN; k0 += 16) {
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
      int m = idx >> 4, kk = idx & 15;                 // A[m][k] row-major (bf16)
      As[idx] = (mt + m < M) ? __float2bfloat16(x[(int64_t)(mt + m) * IN + k0 + kk]) : (__nv_bfloat16)0;
      int n = idx >> 4, kb = idx & 15;                 // B[k][n] col-major: Bs[kb + n*16] = W[nt+n, k0+kb]
      int o = nt + n; float w = 0.f;
      if (o < OUT) {
        uint8_t byte = packed[(int64_t)o * in2 + ((k0 + kb) >> 1)];
        unsigned nib = ((k0 + kb) & 1) ? (byte >> 4) : (byte & 0xF);
        w = fp4_to_float(nib) * fp8e4m3_to_float(scale[(int64_t)o * gg + (k0 + kb) / 16]) / gscale;
      }
      Bs[kb + n * 16] = __float2bfloat16(w);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
    wmma::load_matrix_sync(af, As, 16);
    wmma::load_matrix_sync(bf, Bs, 16);
    wmma::mma_sync(cf, af, bf, cf);
    __syncthreads();
  }
  wmma::store_matrix_sync(Cs, cf, 16, wmma::mem_row_major);
  for (int idx = threadIdx.x; idx < 256; idx += 32) {
    int m = idx >> 4, n = idx & 15;
    if (mt + m < M && nt + n < OUT) y[(int64_t)(mt + m) * OUT + nt + n] = Cs[idx];
  }
}

void gemm_nvfp4(const float* x, const uint8_t* packed, const uint8_t* scale, float gscale,
                float* y, int M, int64_t out, int64_t in, int group) {
  int in2 = (int)(in / 2);
  int gg = (int)(in / group);
  // Tensor-core WMMA path (WIP): correct at bf16 precision but currently slower
  // than the staged f32 kernel because the B-staging does naive scalar FP4 dequant
  // with per-byte global reads. Needs vectorized (uint4) weight loads + hardware
  // e2m1x2 decode + wider K-tiling to feed the tensor cores at bandwidth. Gated off
  // until then (set LB_WMMA=1 to try). This is the pinned next step toward 500+.
  if (M >= 16 && group == 16 && (in % 16) == 0 && getenv("LB_WMMA")) {
    dim3 grid((unsigned)((out + 15) / 16), (unsigned)((M + 15) / 16));
    gemm_nvfp4_wmma_kernel<<<grid, 32>>>(x, packed, scale, gscale, y, M, (int)out, (int)in, in2, gg);
    return;
  }
  if (M == 1 && group == 16 && (in % 1024) == 0) {   // fast decode GEMV path
    const int W = 8;
    int grid = (int)((out + W - 1) / W);
    gemv_nvfp4_m1_kernel<W><<<grid, W * 32>>>(x, packed, scale, gscale, y, (int)out, (int)in, in2, gg);
    return;
  }
  if (M >= 2 && M <= STAGEM && group == 16 && (in % 1024) == 0) {   // shared-staged batch
    const int W = 8;
    int grid = (int)((out + W - 1) / W);
    size_t shmem = (size_t)M * 32 * 33 * sizeof(float);
    static int cfg = 0;
    if (!cfg) { cudaFuncSetAttribute(gemm_nvfp4_stage_kernel<W>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cfg = 1; }
    gemm_nvfp4_stage_kernel<W><<<grid, W * 32, shmem>>>(x, packed, scale, gscale, y, M, (int)out, (int)in, in2, gg);
    return;
  }
  if (M >= 2 && M <= MAXB && group == 16 && (in % 1024) == 0) {   // wider batch (register tiling)
    const int W = 8;
    int grid = (int)((out + W - 1) / W);
    gemm_nvfp4_smallM_kernel<W><<<grid, W * 32>>>(x, packed, scale, gscale, y, M, (int)out, (int)in, in2, gg);
    return;
  }
  dim3 block(64, 4);
  dim3 grid((unsigned)((out + block.x - 1) / block.x), (unsigned)((M + block.y - 1) / block.y));
  gemm_nvfp4_kernel<<<grid, block>>>(x, packed, scale, gscale, y, M, (int)out, (int)in,
                                     in2, gg, group);
}

}  // namespace lb
