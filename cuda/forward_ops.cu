#include "forward.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdlib>
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
// Bandwidth-optimal M=1 (decode) BF16 GEMV: one warp per output row, K tiles of
// 1024. No shared staging / no barriers — activations (tiny, L2-resident) are read
// straight from global, and each K-tile issues all four 128-bit weight loads before
// the FMAs so the load latency is hidden (bf16 has little ALU to hide it otherwise).
template <int WARPS>
__global__ void gemv_bf16_m1_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                    float* __restrict__ y, int OUT, int IN) {
  int lane = threadIdx.x & 31;
  int o = blockIdx.x * WARPS + (threadIdx.x >> 5);
  if (o >= OUT) return;
  const uint16_t* wr = w + (int64_t)o * IN;
  int ntiles = IN / 1024;
  float acc = 0.f;
  uint4 wk[4];                                                  // current tile's 32 weights
  { const uint4* wp = reinterpret_cast<const uint4*>(wr + lane * 32);
    #pragma unroll
    for (int u = 0; u < 4; ++u) wk[u] = wp[u]; }                // prefetch tile 0
  for (int k = 0; k < ntiles; ++k) {
    uint4 nx[4];                                                // prefetch tile k+1 while computing tile k
    if (k + 1 < ntiles) { const uint4* np = reinterpret_cast<const uint4*>(wr + (k + 1) * 1024 + lane * 32);
      #pragma unroll
      for (int u = 0; u < 4; ++u) nx[u] = np[u]; }
    const float4* xp = reinterpret_cast<const float4*>(x + k * 1024 + lane * 32);
    float4 a[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u) a[u] = xp[u];
    const float* av = &a[0].x;
    #pragma unroll
    for (int u = 0; u < 4; ++u) { const uint16_t* hh = (const uint16_t*)&wk[u];
      #pragma unroll
      for (int t = 0; t < 8; ++t) acc = fmaf(av[u * 8 + t], bf16f(hh[t]), acc); }
    #pragma unroll
    for (int u = 0; u < 4; ++u) wk[u] = nx[u];
  }
  for (int d = 16; d; d >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, d);
  if (lane == 0) y[o] = acc;
}

// Weight-stationary batched BF16 GEMM (2..MAXB rows): weight row loaded once,
// reused across M activation vectors. Bandwidth-bound in the weights.
static const int MAXB_BF = 32;
template <int WARPS>
__global__ void gemm_bf16_smallM_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                        float* __restrict__ y, int M, int OUT, int IN) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int o = blockIdx.x * WARPS + warp;
  if (o >= OUT) return;
  const uint16_t* wr = w + (int64_t)o * IN;
  float acc[MAXB_BF]; for (int m = 0; m < M; ++m) acc[m] = 0.f;
  for (int k0 = 0; k0 < IN; k0 += 1024) {
    int col = k0 + lane * 32;
    const uint4* wp = reinterpret_cast<const uint4*>(wr + col);
    float wv[32];
    #pragma unroll
    for (int u = 0; u < 4; ++u) { uint4 pk = wp[u]; const uint16_t* hh = reinterpret_cast<const uint16_t*>(&pk);
      #pragma unroll
      for (int t = 0; t < 8; ++t) wv[u * 8 + t] = bf16f(hh[t]); }
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

// Tensor-core (bf16 WMMA) batched BF16 GEMM — same weight-stationary structure as
// the NVFP4 wmma path but the weights are already bf16 (no dequant). Lifts the M>32
// cliff for the GDN in-projections and lm_head.
namespace wmma_bf = nvcuda::wmma;
static const int NWARP_BF = 8, BMBF = 64;
__global__ void gemm_bf16_wmma_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                      float* __restrict__ y, int M, int OUT, int IN) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int mt = blockIdx.y * BMBF, nt = (blockIdx.x * NWARP_BF + warp) * 16;
  __shared__ __nv_bfloat16 As[BMBF * 64];
  __shared__ __nv_bfloat16 Bs[NWARP_BF][64 * 16];
  __shared__ float Cs[NWARP_BF][16 * 16];
  const int MF = BMBF / 16;
  wmma_bf::fragment<wmma_bf::accumulator, 16, 16, 16, float> cf[MF];
  #pragma unroll
  for (int j = 0; j < MF; ++j) wmma_bf::fill_fragment(cf[j], 0.f);
  for (int k0 = 0; k0 < IN; k0 += 64) {
    for (int idx = threadIdx.x; idx < BMBF * 64; idx += NWARP_BF * 32) {
      int m = idx >> 6, kk = idx & 63;
      As[idx] = (mt + m < M) ? __float2bfloat16(x[(int64_t)(mt + m) * IN + k0 + kk]) : (__nv_bfloat16)0;
    }
    for (int i = lane; i < 16 * 64; i += 32) {          // warp stages its bf16 B tile (col-major)
      int n = i >> 6, kk = i & 63, o = nt + n;
      Bs[warp][kk + n * 64] = (o < OUT) ? *reinterpret_cast<const __nv_bfloat16*>(&w[(int64_t)o * IN + k0 + kk]) : (__nv_bfloat16)0;
    }
    __syncthreads();
    #pragma unroll
    for (int ks = 0; ks < 64; ks += 16) {
      wmma_bf::fragment<wmma_bf::matrix_b, 16, 16, 16, __nv_bfloat16, wmma_bf::col_major> bf;
      wmma_bf::load_matrix_sync(bf, Bs[warp] + ks, 64);
      #pragma unroll
      for (int j = 0; j < MF; ++j) {
        wmma_bf::fragment<wmma_bf::matrix_a, 16, 16, 16, __nv_bfloat16, wmma_bf::row_major> af;
        wmma_bf::load_matrix_sync(af, As + j * 16 * 64 + ks, 64);
        wmma_bf::mma_sync(cf[j], af, bf, cf[j]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int j = 0; j < MF; ++j) {
    wmma_bf::store_matrix_sync(Cs[warp], cf[j], 16, wmma_bf::mem_row_major);
    for (int idx = lane; idx < 256; idx += 32) {
      int m = idx >> 4, n = idx & 15, row = mt + j * 16 + m;
      if (row < M && nt + n < OUT) y[(int64_t)row * OUT + nt + n] = Cs[warp][idx];
    }
    __syncwarp();
  }
}

void gemm_bf16(const float* x, const uint16_t* w, float* y, int M, int OUT, int IN) {
  if (M >= 16 && (IN % 64) == 0 && !getenv("LB_NOWMMA")) {   // tensor-core batched path
    dim3 grid((OUT + NWARP_BF * 16 - 1) / (NWARP_BF * 16), (M + BMBF - 1) / BMBF);
    gemm_bf16_wmma_kernel<<<grid, NWARP_BF * 32>>>(x, w, y, M, OUT, IN);
    return;
  }
  if ((IN % 1024) == 0 && M >= 1 && M <= MAXB_BF) {    // weight-stationary GEMV / small batch
    const int W = 8;
    int grid = (OUT + W - 1) / W;
    if (M == 1) gemv_bf16_m1_kernel<W><<<grid, W * 32>>>(x, w, y, OUT, IN);
    else gemm_bf16_smallM_kernel<W><<<grid, W * 32>>>(x, w, y, M, OUT, IN);
    return;
  }
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

// interleaved M-RoPE: rotary pair i uses pos3d[(i%3)*T + t].
__global__ void rope_mrope_kernel(float* __restrict__ t_arr, const int* __restrict__ pos3d,
                                  int T, int NHEADS, int HD, int rot, float theta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // rotary pair index
  int h = blockIdx.y, t = blockIdx.z;
  int half = rot / 2;
  if (i >= half || h >= NHEADS || t >= T) return;
  float* base = t_arr + (((int64_t)t * NHEADS + h) * HD);
  int axis = i % 3;                                // 0/1/2 = t/h/w  (== mrope_section [11,11,10])
  float freq = __powf(theta, -2.f * i / rot);
  float ang = pos3d[axis * T + t] * freq;
  float c = __cosf(ang), s = __sinf(ang);
  float a = base[i], b = base[i + half];
  base[i] = a * c - b * s;
  base[i + half] = b * c + a * s;
}
void rope_mrope(float* q, float* k, const int* pos3d, int T, int NH, int NKV, int HD,
                int rot, float theta) {
  int half = rot / 2;
  dim3 block(32);
  rope_mrope_kernel<<<dim3((half + 31) / 32, NH, T), block>>>(q, pos3d, T, NH, HD, rot, theta);
  rope_mrope_kernel<<<dim3((half + 31) / 32, NKV, T), block>>>(k, pos3d, T, NKV, HD, rot, theta);
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
  static int* d_idx = nullptr;                 // persistent — avoid malloc/free (+sync) per call
  if (!d_idx) cudaMalloc(&d_idx, sizeof(int));
  int block = 256;
  argmax_kernel<<<1, block, block * (sizeof(float) + sizeof(int))>>>(x_dev, n, d_idx);
  int h_idx; cudaMemcpy(&h_idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost);
  return h_idx;
}

}  // namespace lb
