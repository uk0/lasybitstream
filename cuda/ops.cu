#include "ops.hpp"
#include <cuda_runtime.h>
#include <algorithm>

namespace lb {

// One block per row; f32 sum-of-squares reduction in shared memory.
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               float eps, float* __restrict__ y, int H) {
  extern __shared__ float sh[];
  const float* xr = x + (int64_t)blockIdx.x * H;
  float* yr = y + (int64_t)blockIdx.x * H;
  float local = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = xr[i];
    local += v * v;
  }
  sh[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(sh[0] / H + eps);
  // Gemma-style RMSNorm: scale by (1 + weight), matching vLLM's Qwen3_5RMSNorm.
  for (int i = threadIdx.x; i < H; i += blockDim.x) yr[i] = xr[i] * inv * (1.f + w[i]);
}

void rmsnorm(const float* x, const float* w, float eps, float* y, int M, int H) {
  int block = 256;  // power of two for the tree reduction
  rmsnorm_kernel<<<M, block, block * sizeof(float)>>>(x, w, eps, y, H);
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u,
                              float* __restrict__ o, int64_t n) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    float x = g[i];
    float silu = x / (1.f + expf(-x));
    o[i] = silu * u[i];
  }
}

void swiglu(const float* g, const float* u, float* o, int64_t n) {
  int block = 256;
  int grid = (int)std::min<int64_t>(65535, (n + block - 1) / block);
  swiglu_kernel<<<grid, block>>>(g, u, o, n);
}

}  // namespace lb
