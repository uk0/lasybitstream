// Verify the GDN gated-delta-net recurrence kernel against the fla-matching
// reference (scripts/gdn_ref.py -> test/gdn_*.f32).
#include "gdn.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

static std::vector<float> load(const char* path, size_t n) {
  std::vector<float> v(n);
  std::ifstream f(path, std::ios::binary);
  if (!f) { printf("missing %s (run scripts/gdn_ref.py)\n", path); std::exit(2); }
  f.read((char*)v.data(), (std::streamsize)n * 4);
  return v;
}

int main() {
  const int T = 8, HK = 16, HV = 48, DK = 128, DV = 128;
  const float scale = 1.0f;
  auto q = load("test/gdn_q.f32", (size_t)T * HK * DK);
  auto k = load("test/gdn_k.f32", (size_t)T * HK * DK);
  auto v = load("test/gdn_v.f32", (size_t)T * HV * DV);
  auto g = load("test/gdn_g.f32", (size_t)T * HV);
  auto beta = load("test/gdn_beta.f32", (size_t)T * HV);
  auto ref = load("test/gdn_out.f32", (size_t)T * HV * DV);

  float *dq, *dk, *dv, *dg, *db, *dout, *dstate;
  cudaMalloc(&dq, q.size() * 4); cudaMalloc(&dk, k.size() * 4); cudaMalloc(&dv, v.size() * 4);
  cudaMalloc(&dg, g.size() * 4); cudaMalloc(&db, beta.size() * 4);
  cudaMalloc(&dout, ref.size() * 4);
  cudaMalloc(&dstate, (size_t)HV * DV * DK * 4);
  cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(db, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice);

  lb::gdn_recurrence(dq, dk, dv, dg, db, scale, dout, dstate, T, HK, HV, DK, DV);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }

  std::vector<float> got(ref.size());
  cudaMemcpy(got.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost);

  double maxabs = 0, maxrel = 0, sum = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    double d = std::fabs((double)got[i] - ref[i]);
    maxabs = std::max(maxabs, d); sum += d;
    maxrel = std::max(maxrel, d / (std::fabs((double)ref[i]) + 1e-4));
  }
  printf("GDN recurrence out[%d,%d,%d] vs ref: max_abs=%.2e mean_abs=%.2e max_rel=%.2e (ref[0]=%.4f got[0]=%.4f)\n",
         T, HV, DV, maxabs, sum / got.size(), maxrel, ref[0], got[0]);
  bool pass = maxrel < 2e-3 && maxabs < 1e-3;
  printf("== %s ==\n", pass ? "PASS — GDN gated-delta-net recurrence matches reference" : "FAIL");
  return pass ? 0 : 1;
}
