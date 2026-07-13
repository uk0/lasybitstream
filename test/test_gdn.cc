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
  bool ok = maxrel < 2e-3 && maxabs < 1e-3;
  printf("recurrence out[%d,%d,%d]: max_abs=%.2e max_rel=%.2e %s\n",
         T, HV, DV, maxabs, maxrel, ok ? "PASS" : "FAIL");

  // gating: g = -exp(A_log)*softplus(a+dt_bias), beta = sigmoid(b)
  {
    std::vector<float> Al(HV), dt(HV), aa((size_t)T * HV), bb((size_t)T * HV);
    std::vector<float> gr((size_t)T * HV), br((size_t)T * HV), gg((size_t)T * HV), bg((size_t)T * HV);
    for (int i = 0; i < HV; ++i) { Al[i] = std::sin(0.1f * i) * 0.5f; dt[i] = std::cos(0.07f * i) * 0.3f; }
    for (int i = 0; i < T * HV; ++i) { aa[i] = std::sin(0.03f * i) * 1.2f; bb[i] = std::cos(0.05f * i) * 1.5f; }
    for (int t = 0; t < T; ++t) for (int h = 0; h < HV; ++h) {
      int id = t * HV + h; float x = aa[id] + dt[h];
      float sp = x > 20.f ? x : std::log1p(std::exp(x));
      gr[id] = -std::exp(Al[h]) * sp; br[id] = 1.f / (1.f + std::exp(-bb[id]));
    }
    float *dAl, *ddt, *da, *db2, *dgo, *dbo;
    cudaMalloc(&dAl, HV * 4); cudaMalloc(&ddt, HV * 4); cudaMalloc(&da, T * HV * 4);
    cudaMalloc(&db2, T * HV * 4); cudaMalloc(&dgo, T * HV * 4); cudaMalloc(&dbo, T * HV * 4);
    cudaMemcpy(dAl, Al.data(), HV * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ddt, dt.data(), HV * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(da, aa.data(), T * HV * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db2, bb.data(), T * HV * 4, cudaMemcpyHostToDevice);
    lb::gdn_gating(dAl, da, db2, ddt, dgo, dbo, T, HV);
    cudaDeviceSynchronize();
    cudaMemcpy(gg.data(), dgo, T * HV * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(bg.data(), dbo, T * HV * 4, cudaMemcpyDeviceToHost);
    double ma = 0; for (int i = 0; i < T * HV; ++i) { ma = std::max(ma, std::fabs((double)gg[i] - gr[i])); ma = std::max(ma, std::fabs((double)bg[i] - br[i])); }
    bool p = ma < 1e-5; ok &= p;
    printf("gating (g,beta): max_abs=%.2e %s\n", ma, p ? "PASS" : "FAIL");
  }

  // rmsnorm_gated: out = rmsnorm(x)*w * silu(z)
  {
    int Mr = 4, Hr = 128;
    std::vector<float> x((size_t)Mr * Hr), w(Hr), z((size_t)Mr * Hr), rr((size_t)Mr * Hr), rg((size_t)Mr * Hr);
    for (int i = 0; i < Mr * Hr; ++i) { x[i] = std::sin(0.02f * i) * 1.1f; z[i] = std::cos(0.015f * i) * 0.9f; }
    for (int i = 0; i < Hr; ++i) w[i] = 0.7f + 0.3f * std::sin(0.05f * i);
    for (int m = 0; m < Mr; ++m) {
      double ss = 0; for (int i = 0; i < Hr; ++i) { float vv = x[m * Hr + i]; ss += (double)vv * vv; }
      float inv = (float)(1.0 / std::sqrt(ss / Hr + 1e-6));
      for (int i = 0; i < Hr; ++i) { float zi = z[m * Hr + i]; float si = zi / (1.f + std::exp(-zi)); rr[m * Hr + i] = x[m * Hr + i] * inv * w[i] * si; }
    }
    float *dx, *dw, *dz, *dr;
    cudaMalloc(&dx, Mr * Hr * 4); cudaMalloc(&dw, Hr * 4); cudaMalloc(&dz, Mr * Hr * 4); cudaMalloc(&dr, Mr * Hr * 4);
    cudaMemcpy(dx, x.data(), Mr * Hr * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), Hr * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dz, z.data(), Mr * Hr * 4, cudaMemcpyHostToDevice);
    lb::rmsnorm_gated(dx, dw, dz, 1e-6f, dr, Mr, Hr);
    cudaDeviceSynchronize();
    cudaMemcpy(rg.data(), dr, Mr * Hr * 4, cudaMemcpyDeviceToHost);
    double ma = 0; for (int i = 0; i < Mr * Hr; ++i) ma = std::max(ma, std::fabs((double)rg[i] - rr[i]));
    bool p = ma < 1e-5; ok &= p;
    printf("rmsnorm_gated: max_abs=%.2e %s\n", ma, p ? "PASS" : "FAIL");
  }

  printf("== %s ==\n", ok ? "PASS — GDN recurrence + gating + gated-norm match references" : "FAIL");
  return ok ? 0 : 1;
}
