#include "gdn.hpp"
#include <cuda_runtime.h>

namespace lb {

// One block per v-head; blockDim.x = head_v threads (thread r owns state row r).
// State S[head_v, head_k] lives in global scratch; k/q of the token are staged in
// shared memory and L2-normalized before the row updates.
__global__ void gdn_recurrence_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, const float* __restrict__ g,
                                      const float* __restrict__ beta, float scale,
                                      float* __restrict__ out, float* __restrict__ state,
                                      int T, int HK, int HV, int DK, int DV,
                                      float* __restrict__ snap) {
  extern __shared__ float sh[];
  float* ksh = sh;            // [DK]
  float* qsh = sh + DK;       // [DK]
  float* red = sh + 2 * DK;   // [blockDim.x] reduction scratch

  int hv = blockIdx.x;
  int r = threadIdx.x;                 // state row (head_v element)
  int kh = hv / (HV / HK);             // GQA: k-head for this v-head
  float* S = state + (int64_t)hv * DV * DK;  // this head's state [DV, DK]

  for (int t = 0; t < T; ++t) {
    const float* kt = k + ((int64_t)t * HK + kh) * DK;
    const float* qt = q + ((int64_t)t * HK + kh) * DK;
    if (r < DK) { ksh[r] = kt[r]; qsh[r] = qt[r]; }
    __syncthreads();

    // L2-normalize k, then q (block reduction over DK).
    for (int pass = 0; pass < 2; ++pass) {
      float* vec = pass == 0 ? ksh : qsh;
      red[r] = (r < DK) ? vec[r] * vec[r] : 0.f;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (r < s) red[r] += red[r + s];
        __syncthreads();
      }
      float inv = rsqrtf(red[0] + 1e-6f);
      __syncthreads();
      if (r < DK) vec[r] = vec[r] * inv * (pass == 1 ? scale : 1.f);
      __syncthreads();
    }

    float gt = expf(g[(int64_t)t * HV + hv]);
    float bt = beta[(int64_t)t * HV + hv];
    float* Srow = S + (int64_t)r * DK;

    float vpred = 0.f;
    for (int c = 0; c < DK; ++c) { Srow[c] *= gt; vpred += Srow[c] * ksh[c]; }
    float vr = (v[((int64_t)t * HV + hv) * DV + r] - vpred) * bt;
    float o = 0.f;
    for (int c = 0; c < DK; ++c) { Srow[c] += vr * ksh[c]; o += Srow[c] * qsh[c]; }
    out[((int64_t)t * HV + hv) * DV + r] = o;
    if (snap) { float* dst = snap + (((int64_t)t * HV + hv) * DV + r) * DK;   // state after token t
      for (int c = 0; c < DK; ++c) dst[c] = Srow[c]; }
    __syncthreads();  // ksh/qsh reused next token
  }
}

void gdn_recurrence(const float* q, const float* k, const float* v, const float* g,
                    const float* beta, float scale, float* out, float* state,
                    int T, int HK, int HV, int DK, int DV, bool reset, float* snap) {
  if (reset) cudaMemset(state, 0, (size_t)HV * DV * DK * sizeof(float));
  int block = DV;  // one thread per state row
  size_t shmem = (2 * DK + block) * sizeof(float);
  gdn_recurrence_kernel<<<HV, block, shmem>>>(q, k, v, g, beta, scale, out, state,
                                              T, HK, HV, DK, DV, snap);
}

// Batched decode: B independent sequences, ONE token each, own state [B,HV,DV,DK].
// grid (HV, B) — one block per (v-head, sequence); blockDim.x = DV (thread r owns state row r).
// The state row S[r,:] (DK floats) is held in REGISTERS across both dependent passes,
// so it hits global memory once (load) + once (store) instead of two read-modify-write
// traversals — GDN state traffic per step is the dominant non-GEMM cost. `DKC` is the
// compile-time head_k so `sr[DKC]` is register-resident (fully unrolled, no local spill).
template <int DKC>
__global__ void gdn_recurrence_batched_reg_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                  const float* __restrict__ v, const float* __restrict__ g,
                                                  const float* __restrict__ beta, float scale,
                                                  float* __restrict__ out, float* __restrict__ state,
                                                  int B, int HK, int HV, int DV) {
  extern __shared__ float sh[];
  float* ksh = sh; float* qsh = sh + DKC; float* red = sh + 2 * DKC;
  int hv = blockIdx.x, bseq = blockIdx.y, r = threadIdx.x;
  int kh = hv / (HV / HK);
  float* Srow = state + (((int64_t)bseq * HV + hv) * DV + r) * DKC;   // this thread's state row
  const float* kt = k + ((int64_t)bseq * HK + kh) * DKC;
  const float* qt = q + ((int64_t)bseq * HK + kh) * DKC;
  if (r < DKC) { ksh[r] = kt[r]; qsh[r] = qt[r]; }
  __syncthreads();
  for (int pass = 0; pass < 2; ++pass) {                              // L2-normalize k, then q
    float* vec = pass == 0 ? ksh : qsh;
    red[r] = (r < DKC) ? vec[r] * vec[r] : 0.f;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (r < s) red[r] += red[r + s]; __syncthreads(); }
    float inv = rsqrtf(red[0] + 1e-6f);
    __syncthreads();
    if (r < DKC) vec[r] = vec[r] * inv * (pass == 1 ? scale : 1.f);
    __syncthreads();
  }
  float gt = expf(g[(int64_t)bseq * HV + hv]);
  float bt = beta[(int64_t)bseq * HV + hv];
  float sr[DKC];                                                      // register-resident state row
  float vpred = 0.f;
#pragma unroll
  for (int c = 0; c < DKC; ++c) { sr[c] = Srow[c] * gt; vpred += sr[c] * ksh[c]; }   // load once + decay + dot
  float vr = (v[((int64_t)bseq * HV + hv) * DV + r] - vpred) * bt;
  float o = 0.f;
#pragma unroll
  for (int c = 0; c < DKC; ++c) { sr[c] += vr * ksh[c]; o += sr[c] * qsh[c]; }        // rank-1 update + readout
#pragma unroll
  for (int c = 0; c < DKC; ++c) Srow[c] = sr[c];                                       // store once
  out[((int64_t)bseq * HV + hv) * DV + r] = o;
}

// Generic fallback (arbitrary DK), streaming the state row through global memory.
__global__ void gdn_recurrence_batched_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                              const float* __restrict__ v, const float* __restrict__ g,
                                              const float* __restrict__ beta, float scale,
                                              float* __restrict__ out, float* __restrict__ state,
                                              int B, int HK, int HV, int DK, int DV) {
  extern __shared__ float sh[];
  float* ksh = sh; float* qsh = sh + DK; float* red = sh + 2 * DK;
  int hv = blockIdx.x, bseq = blockIdx.y, r = threadIdx.x;
  int kh = hv / (HV / HK);
  float* Srow = state + (((int64_t)bseq * HV + hv) * DV + r) * DK;
  const float* kt = k + ((int64_t)bseq * HK + kh) * DK;
  const float* qt = q + ((int64_t)bseq * HK + kh) * DK;
  if (r < DK) { ksh[r] = kt[r]; qsh[r] = qt[r]; }
  __syncthreads();
  for (int pass = 0; pass < 2; ++pass) {
    float* vec = pass == 0 ? ksh : qsh;
    red[r] = (r < DK) ? vec[r] * vec[r] : 0.f;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (r < s) red[r] += red[r + s]; __syncthreads(); }
    float inv = rsqrtf(red[0] + 1e-6f);
    __syncthreads();
    if (r < DK) vec[r] = vec[r] * inv * (pass == 1 ? scale : 1.f);
    __syncthreads();
  }
  float gt = expf(g[(int64_t)bseq * HV + hv]);
  float bt = beta[(int64_t)bseq * HV + hv];
  float vpred = 0.f;
  for (int c = 0; c < DK; ++c) { Srow[c] *= gt; vpred += Srow[c] * ksh[c]; }
  float vr = (v[((int64_t)bseq * HV + hv) * DV + r] - vpred) * bt;
  float o = 0.f;
  for (int c = 0; c < DK; ++c) { Srow[c] += vr * ksh[c]; o += Srow[c] * qsh[c]; }
  out[((int64_t)bseq * HV + hv) * DV + r] = o;
}
void gdn_recurrence_batched(const float* q, const float* k, const float* v, const float* g,
                            const float* beta, float scale, float* out, float* state,
                            int B, int HK, int HV, int DK, int DV) {
  dim3 grid(HV, B);
  int block = DV;
  size_t shmem = (2 * DK + block) * sizeof(float);
  if (DK == 128)   // this model: head_k = head_v = 128 -> register-resident state row
    gdn_recurrence_batched_reg_kernel<128><<<grid, block, shmem>>>(q, k, v, g, beta, scale, out, state, B, HK, HV, DV);
  else
    gdn_recurrence_batched_kernel<<<grid, block, shmem>>>(q, k, v, g, beta, scale, out, state, B, HK, HV, DK, DV);
}

__global__ void gdn_gating_kernel(const float* __restrict__ A_log, const float* __restrict__ a,
                                  const float* __restrict__ b, const float* __restrict__ dt_bias,
                                  float* __restrict__ g_out, float* __restrict__ beta_out,
                                  int T, int HV) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (int64_t)T * HV) return;
  int hv = (int)(idx % HV);
  float x = a[idx] + dt_bias[hv];
  float sp = (x > 20.f) ? x : log1pf(expf(x));       // softplus, stable
  g_out[idx] = -expf(A_log[hv]) * sp;
  beta_out[idx] = 1.f / (1.f + expf(-b[idx]));        // sigmoid
}

void gdn_gating(const float* A_log, const float* a, const float* b, const float* dt_bias,
                float* g_out, float* beta_out, int T, int HV) {
  int block = 256;
  int grid = (int)(((int64_t)T * HV + block - 1) / block);
  gdn_gating_kernel<<<grid, block>>>(A_log, a, b, dt_bias, g_out, beta_out, T, HV);
}

// out = rmsnorm(x)*weight * silu(z), one block per row.
__global__ void rmsnorm_gated_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                     const float* __restrict__ z, float eps,
                                     float* __restrict__ out, int H) {
  extern __shared__ float red[];
  const float* xr = x + (int64_t)blockIdx.x * H;
  const float* zr = z + (int64_t)blockIdx.x * H;
  float* outr = out + (int64_t)blockIdx.x * H;
  float local = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) local += xr[i] * xr[i];
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / H + eps);
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float zi = zr[i];
    float silu = zi / (1.f + expf(-zi));
    outr[i] = xr[i] * inv * w[i] * silu;
  }
}

void rmsnorm_gated(const float* x, const float* w, const float* z, float eps,
                   float* out, int M, int H) {
  int block = 128;
  rmsnorm_gated_kernel<<<M, block, block * sizeof(float)>>>(x, w, z, eps, out, H);
}

}  // namespace lb
