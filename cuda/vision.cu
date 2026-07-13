#include "vision.hpp"
#include <cuda_runtime.h>
#include <cfloat>

namespace lb {

// ---- LayerNorm (mean-centering) with bias ----
__global__ void layernorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                 const float* __restrict__ b, float eps, float* __restrict__ y, int H) {
  extern __shared__ float red[];
  const float* xr = x + (int64_t)blockIdx.x * H;
  float* yr = y + (int64_t)blockIdx.x * H;
  float s = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) s += xr[i];
  red[threadIdx.x] = s; __syncthreads();
  for (int k = blockDim.x >> 1; k > 0; k >>= 1) { if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k]; __syncthreads(); }
  float mean = red[0] / H; __syncthreads();
  float v = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) { float d = xr[i] - mean; v += d * d; }
  red[threadIdx.x] = v; __syncthreads();
  for (int k = blockDim.x >> 1; k > 0; k >>= 1) { if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k]; __syncthreads(); }
  float inv = rsqrtf(red[0] / H + eps);
  for (int i = threadIdx.x; i < H; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}
void layernorm(const float* x, const float* w, const float* b, float eps, float* y, int M, int H) {
  int block = 256;
  layernorm_kernel<<<M, block, block * sizeof(float)>>>(x, w, b, eps, y, H);
}

// ---- GELU variants ----
__global__ void gelu_tanh_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  y[i] = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
void gelu_tanh(const float* x, float* y, int64_t n) {
  int block = 256; int64_t grid = (n + block - 1) / block;
  gelu_tanh_kernel<<<grid, block>>>(x, y, n);
}
__global__ void gelu_erf_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  y[i] = 0.5f * v * (1.f + erff(v * 0.7071067811865476f));
}
void gelu_erf(const float* x, float* y, int64_t n) {
  int block = 256; int64_t grid = (n + block - 1) / block;
  gelu_erf_kernel<<<grid, block>>>(x, y, n);
}

// ---- bias add ----
__global__ void add_bias_kernel(float* __restrict__ y, const float* __restrict__ b, int M, int N) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y;
  if (j < N && m < M) y[(int64_t)m * N + j] += b[j];
}
void add_bias(float* y, const float* b, int M, int N) {
  dim3 block(256), grid((N + 255) / 256, M);
  add_bias_kernel<<<grid, block>>>(y, b, M, N);
}

// ---- vision 2D RoPE ----
// one thread per (s, head, i) with i in [0, HD/2); rotate_half splits HD into halves.
__global__ void rope_vision_kernel(float* __restrict__ t, const float* __restrict__ cos,
                                   const float* __restrict__ sin, int S, int NH, int HD) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // [0, HD/2)
  int h = blockIdx.y, s = blockIdx.z;
  int half = HD / 2;
  if (i >= half || h >= NH || s >= S) return;
  float* base = t + (((int64_t)s * NH + h) * HD);
  const float* c = cos + (int64_t)s * HD;
  const float* sn = sin + (int64_t)s * HD;
  float a = base[i], b = base[i + half];
  // rotate_half(x) = [-x[half:], x[:half]]  ->  out[i]=a*c[i]-b*sn[i]; out[i+half]=b*c[i+half]+a*sn[i+half]
  base[i] = a * c[i] + (-b) * sn[i];
  base[i + half] = b * c[i + half] + a * sn[i + half];
}
void rope_vision(float* q, float* k, const float* cos, const float* sin, int S, int NH, int HD) {
  dim3 block(32), grid((HD / 2 + 31) / 32, NH, S);
  rope_vision_kernel<<<grid, block>>>(q, cos, sin, S, NH, HD);
  rope_vision_kernel<<<grid, block>>>(k, cos, sin, S, NH, HD);
}

// ---- full bidirectional MHA ----
// one block per (head, query s); threads == HD; online softmax over all S keys.
__global__ void vision_attn_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                   const float* __restrict__ v, float* __restrict__ out,
                                   int S, int NH, int HD) {
  int h = blockIdx.x, sq = blockIdx.y;
  if (h >= NH || sq >= S) return;
  int lane = threadIdx.x;
  extern __shared__ float sh[];
  float* qsh = sh; float* red = sh + HD;
  qsh[lane] = q[(((int64_t)sq * NH + h) * HD) + lane];
  __syncthreads();
  float scale = rsqrtf((float)HD);
  float m = -FLT_MAX, l = 0.f, acc = 0.f;
  for (int sk = 0; sk < S; ++sk) {
    const float* kk = k + (((int64_t)sk * NH + h) * HD);
    red[lane] = qsh[lane] * kk[lane];
    __syncthreads();
    // reduction that is correct for non-power-of-2 HD (e.g. 72): start at the
    // largest power of two < HD and guard lane+s < HD.
    int s0 = 1; while (s0 < HD) s0 <<= 1; s0 >>= 1;
    for (int s = s0; s > 0; s >>= 1) { if (lane < s && lane + s < HD) red[lane] += red[lane + s]; __syncthreads(); }
    float score = red[0] * scale;
    __syncthreads();
    float mn = fmaxf(m, score), corr = __expf(m - mn), p = __expf(score - mn);
    l = l * corr + p;
    acc = acc * corr + p * v[(((int64_t)sk * NH + h) * HD) + lane];
    m = mn;
  }
  out[(((int64_t)sq * NH + h) * HD) + lane] = acc / l;
}
void vision_attention(const float* q, const float* k, const float* v, float* out, int S, int NH, int HD) {
  dim3 block(HD), grid(NH, S);
  vision_attn_kernel<<<grid, block, (HD + HD) * sizeof(float)>>>(q, k, v, out, S, NH, HD);
}

}  // namespace lb
