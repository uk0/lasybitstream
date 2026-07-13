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
                                      int T, int HK, int HV, int DK, int DV) {
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
    __syncthreads();  // ksh/qsh reused next token
  }
}

void gdn_recurrence(const float* q, const float* k, const float* v, const float* g,
                    const float* beta, float scale, float* out, float* state,
                    int T, int HK, int HV, int DK, int DV) {
  cudaMemset(state, 0, (size_t)HV * DV * DK * sizeof(float));
  int block = DV;  // one thread per state row
  size_t shmem = (2 * DK + block) * sizeof(float);
  gdn_recurrence_kernel<<<HV, block, shmem>>>(q, k, v, g, beta, scale, out, state,
                                              T, HK, HV, DK, DV);
}

}  // namespace lb
