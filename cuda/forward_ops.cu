#include "forward.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

namespace lb {

static __device__ __forceinline__ float bf16f(uint16_t h) {
  __nv_bfloat16 b; *reinterpret_cast<uint16_t*>(&b) = h; return __bfloat162float(b);
}
static __device__ __forceinline__ float siluf(float x) { return x / (1.f + __expf(-x)); }

// ---- BF16 dense GEMM: y[m,o] = sum_i x[m,i] * w[o,i] ----
__global__ void gemm_bf16_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                 float* __restrict__ y, int M, int OUT, int IN) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y;
  if (o >= OUT || m >= M) return;
  const float* xr = x + (int64_t)m * IN;
  const uint16_t* wr = w + (int64_t)o * IN;
  float acc = 0.f;
  for (int i = 0; i < IN; ++i) acc += xr[i] * bf16f(wr[i]);
  y[(int64_t)m * OUT + o] = acc;
}
void gemm_bf16(const float* x, const uint16_t* w, float* y, int M, int OUT, int IN) {
  dim3 block(256), grid((OUT + 255) / 256, M);
  gemm_bf16_kernel<<<grid, block>>>(x, w, y, M, OUT, IN);
}

// ---- embedding gather ----
__global__ void embed_gather_kernel(const uint16_t* __restrict__ e, const int* __restrict__ ids,
                                    float* __restrict__ y, int T, int H) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int t = blockIdx.y;
  if (i >= H || t >= T) return;
  y[(int64_t)t * H + i] = bf16f(e[(int64_t)ids[t] * H + i]);
}
void embed_gather(const uint16_t* e, const int* ids, float* y, int T, int H) {
  dim3 block(256), grid((H + 255) / 256, T);
  embed_gather_kernel<<<grid, block>>>(e, ids, y, T, H);
}

// ---- depthwise causal conv1d + SiLU ----
__global__ void conv1d_silu_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                   float* __restrict__ out, int T, int C, int K) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int t = blockIdx.y;
  if (c >= C || t >= T) return;
  float acc = 0.f;
  for (int j = 0; j < K; ++j) {
    int ti = t - (K - 1) + j;
    if (ti >= 0) acc += bf16f(w[(int64_t)c * K + j]) * x[(int64_t)ti * C + c];
  }
  out[(int64_t)t * C + c] = siluf(acc);
}
void causal_conv1d_silu(const float* x, const uint16_t* w, float* out, int T, int C, int K) {
  dim3 block(256), grid((C + 255) / 256, T);
  conv1d_silu_kernel<<<grid, block>>>(x, w, out, T, C, K);
}

// ---- partial NeoX RoPE (in-place) ----
__global__ void rope_kernel(float* __restrict__ t_arr, const int* __restrict__ pos,
                            int T, int NHEADS, int HD, int rot, float theta) {
  // one thread per (token, head, i) with i in [0, rot/2)
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // rotary pair index
  int h = blockIdx.y;                              // head
  int t = blockIdx.z;                              // token
  int half = rot / 2;
  if (i >= half || h >= NHEADS || t >= T) return;
  float* base = t_arr + (((int64_t)t * NHEADS + h) * HD);
  float freq = __powf(theta, -2.f * i / rot);
  float ang = pos[t] * freq;
  float c = __cosf(ang), s = __sinf(ang);
  float a = base[i], b = base[i + half];
  base[i] = a * c - b * s;
  base[i + half] = b * c + a * s;
}
void rope_partial(float* q, float* k, const int* pos, int T, int NH, int NKV, int HD,
                  int rot, float theta) {
  int half = rot / 2;
  dim3 block(32);
  dim3 gq((half + 31) / 32, NH, T);
  rope_kernel<<<gq, block>>>(q, pos, T, NH, HD, rot, theta);
  dim3 gk((half + 31) / 32, NKV, T);
  rope_kernel<<<gk, block>>>(k, pos, T, NKV, HD, rot, theta);
}

// ---- GQA causal attention ----
// one block per (head, query token); threads cooperate over HD / keys.
__global__ void attention_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                 const float* __restrict__ v, float* __restrict__ out,
                                 int T, int NH, int NKV, int HD) {
  int h = blockIdx.x;        // query head
  int tq = blockIdx.y;       // query token
  if (h >= NH || tq >= T) return;
  int kvh = h / (NH / NKV);
  int lane = threadIdx.x;    // 0..HD-1 (blockDim.x == HD)
  extern __shared__ float sh[];
  float* qsh = sh;           // [HD]
  float* red = sh + HD;      // [blockDim.x]
  const float* qv = q + (((int64_t)tq * NH + h) * HD);
  qsh[lane] = qv[lane];
  __syncthreads();
  float scale = rsqrtf((float)HD);

  // online softmax over keys 0..tq
  float m = -FLT_MAX, l = 0.f, acc = 0.f;   // acc = weighted sum for this lane's dim
  for (int tk = 0; tk <= tq; ++tk) {
    const float* kv = k + (((int64_t)tk * NKV + kvh) * HD);
    // dot(q,k) via block reduction
    red[lane] = qsh[lane] * kv[lane];
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (lane < s) red[lane] += red[lane + s];
      __syncthreads();
    }
    float score = red[0] * scale;
    __syncthreads();
    float mn = fmaxf(m, score);
    float corr = __expf(m - mn);
    float p = __expf(score - mn);
    l = l * corr + p;
    acc = acc * corr + p * v[(((int64_t)tk * NKV + kvh) * HD) + lane];
    m = mn;
  }
  out[(((int64_t)tq * NH + h) * HD) + lane] = acc / l;
}
void attention_gqa(const float* q, const float* k, const float* v, float* out,
                   int T, int NH, int NKV, int HD) {
  dim3 block(HD), grid(NH, T);
  size_t shmem = (HD + HD) * sizeof(float);
  attention_kernel<<<grid, block, shmem>>>(q, k, v, out, T, NH, NKV, HD);
}

// ---- cached (incremental) GQA attention ----
// one block per (head, new-query i); threads == HD, online softmax over the cache.
__global__ void attention_cached_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                                        const float* __restrict__ vc, float* __restrict__ out,
                                        int T, int NH, int NKV, int HD, int start_pos) {
  int h = blockIdx.x;        // query head
  int i = blockIdx.y;        // new-query index [0,T)
  if (h >= NH || i >= T) return;
  int kvh = h / (NH / NKV);
  int lane = threadIdx.x;    // 0..HD-1
  extern __shared__ float sh[];
  float* qsh = sh; float* red = sh + HD;
  qsh[lane] = q[(((int64_t)i * NH + h) * HD) + lane];
  __syncthreads();
  float scale = rsqrtf((float)HD);
  int limit = start_pos + i;   // inclusive causal bound in the cache
  float m = -FLT_MAX, l = 0.f, acc = 0.f;
  for (int tk = 0; tk <= limit; ++tk) {
    const float* kv = kc + (((int64_t)tk * NKV + kvh) * HD);
    red[lane] = qsh[lane] * kv[lane];
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (lane < s) red[lane] += red[lane + s]; __syncthreads(); }
    float score = red[0] * scale;
    __syncthreads();
    float mn = fmaxf(m, score), corr = __expf(m - mn), p = __expf(score - mn);
    l = l * corr + p;
    acc = acc * corr + p * vc[(((int64_t)tk * NKV + kvh) * HD) + lane];
    m = mn;
  }
  out[(((int64_t)i * NH + h) * HD) + lane] = acc / l;
}
void attention_cached(const float* q, const float* kc, const float* vc, float* out,
                      int T, int NH, int NKV, int HD, int start_pos) {
  dim3 block(HD), grid(NH, T);
  attention_cached_kernel<<<grid, block, (HD + HD) * sizeof(float)>>>(q, kc, vc, out, T, NH, NKV, HD, start_pos);
}

// ---- depthwise causal conv1d + SiLU with carried state ----
// one thread per channel, sequential over the T new tokens, sliding a K-1 window.
__global__ void conv1d_state_kernel(const float* __restrict__ x, float* __restrict__ state,
                                    const uint16_t* __restrict__ w, float* __restrict__ out,
                                    int T, int C, int K, int first) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float hist[8];                                  // K-1 previous inputs, oldest..newest
  for (int m = 0; m < K - 1; ++m) hist[m] = first ? 0.f : state[(int64_t)c * (K - 1) + m];
  for (int t = 0; t < T; ++t) {
    float cur = x[(int64_t)t * C + c];
    float acc = bf16f(w[(int64_t)c * K + (K - 1)]) * cur;
    for (int j = 0; j < K - 1; ++j) acc += bf16f(w[(int64_t)c * K + j]) * hist[j];
    out[(int64_t)t * C + c] = siluf(acc);
    for (int m = 0; m < K - 2; ++m) hist[m] = hist[m + 1];
    hist[K - 2] = cur;
  }
  for (int m = 0; m < K - 1; ++m) state[(int64_t)c * (K - 1) + m] = hist[m];
}
void causal_conv1d_state_silu(const float* x, float* state, const uint16_t* w, float* out,
                              int T, int C, int K, bool first) {
  int block = 256, grid = (C + block - 1) / block;
  conv1d_state_kernel<<<grid, block>>>(x, state, w, out, T, C, K, first ? 1 : 0);
}

// ---- per-head q/gate split ----
__global__ void split_qgate_kernel(const float* __restrict__ qg, float* __restrict__ q,
                                   float* __restrict__ gate, int T, int NH, int HD) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y, t = blockIdx.z;
  if (d >= HD || h >= NH || t >= T) return;
  int64_t src = (((int64_t)t * NH + h) * 2 * HD);
  int64_t dst = (((int64_t)t * NH + h) * HD);
  q[dst + d] = qg[src + d];
  gate[dst + d] = qg[src + HD + d];
}
void split_qgate(const float* qg, float* q, float* gate, int T, int NH, int HD) {
  dim3 block(64), grid((HD + 63) / 64, NH, T);
  split_qgate_kernel<<<grid, block>>>(qg, q, gate, T, NH, HD);
}

// ---- sigmoid output gate ----
__global__ void mul_sigmoid_kernel(const float* __restrict__ x, const float* __restrict__ gate,
                                   float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = x[i] * (1.f / (1.f + __expf(-gate[i])));   // sigmoid(gate) * x
}
void mul_sigmoid_gate(const float* x, const float* gate, float* out, int n) {
  int block = 256, grid = (n + block - 1) / block;
  mul_sigmoid_kernel<<<grid, block>>>(x, gate, out, n);
}

// ---- residual add (a += b) ----
__global__ void add_kernel(float* __restrict__ a, const float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}
void add_inplace(float* a, const float* b, int n) {
  int block = 256, grid = (n + block - 1) / block;
  add_kernel<<<grid, block>>>(a, b, n);
}

// ---- argmax (single block) ----
__global__ void argmax_kernel(const float* __restrict__ x, int n, int* __restrict__ idx) {
  extern __shared__ float sh[];
  int* si = (int*)(sh + blockDim.x);
  float best = -FLT_MAX; int bi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    if (x[i] > best) { best = x[i]; bi = i; }
  sh[threadIdx.x] = best; si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s && sh[threadIdx.x + s] > sh[threadIdx.x]) {
      sh[threadIdx.x] = sh[threadIdx.x + s]; si[threadIdx.x] = si[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *idx = si[0];
}
int argmax(const float* x_dev, int n) {
  int* d_idx; cudaMalloc(&d_idx, sizeof(int));
  int block = 256;
  argmax_kernel<<<1, block, block * (sizeof(float) + sizeof(int))>>>(x_dev, n, d_idx);
  int h_idx; cudaMemcpy(&h_idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_idx);
  return h_idx;
}

}  // namespace lb
