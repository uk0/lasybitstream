// Verify RMSNorm and SwiGLU against inline CPU references.
#include "ops.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

static double cmp(const std::vector<float>& a, const std::vector<float>& b, double& maxrel) {
  double maxabs = 0;
  maxrel = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::fabs((double)a[i] - b[i]);
    maxabs = std::max(maxabs, d);
    maxrel = std::max(maxrel, d / (std::fabs((double)b[i]) + 1e-6));
  }
  return maxabs;
}

int main() {
  bool ok = true;

  // RMSNorm
  {
    int M = 2, H = 5120;
    float eps = 1e-6f;
    std::vector<float> x((size_t)M * H), w(H), yref((size_t)M * H), ygot((size_t)M * H);
    for (int i = 0; i < M * H; ++i) x[i] = std::sin(0.01f * i + 1.f) * 1.3f;
    for (int i = 0; i < H; ++i) w[i] = 0.5f + 0.5f * std::cos(0.003f * i);
    for (int m = 0; m < M; ++m) {
      double ss = 0;
      for (int i = 0; i < H; ++i) { float v = x[m * H + i]; ss += (double)v * v; }
      float inv = (float)(1.0 / std::sqrt(ss / H + eps));
      for (int i = 0; i < H; ++i) yref[m * H + i] = x[m * H + i] * inv * w[i];
    }
    float *dx, *dw, *dy;
    cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dw, w.size() * 4); cudaMalloc(&dy, x.size() * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    lb::rmsnorm(dx, dw, eps, dy, M, H);
    cudaDeviceSynchronize();
    cudaMemcpy(ygot.data(), dy, x.size() * 4, cudaMemcpyDeviceToHost);
    double mr, ma = cmp(ygot, yref, mr);
    bool p = ma < 1e-4 && mr < 1e-3;
    printf("RMSNorm[%d,%d]: max_abs=%.2e max_rel=%.2e %s\n", M, H, ma, mr, p ? "PASS" : "FAIL");
    ok &= p; cudaFree(dx); cudaFree(dw); cudaFree(dy);
  }

  // SwiGLU
  {
    int n = 4096;
    std::vector<float> g(n), u(n), oref(n), ogot(n);
    for (int i = 0; i < n; ++i) { g[i] = std::sin(0.02f * i) * 3.f; u[i] = std::cos(0.017f * i) * 1.5f; }
    for (int i = 0; i < n; ++i) { float x = g[i]; float s = x / (1.f + std::exp(-x)); oref[i] = s * u[i]; }
    float *dg, *du, *doo;
    cudaMalloc(&dg, n * 4); cudaMalloc(&du, n * 4); cudaMalloc(&doo, n * 4);
    cudaMemcpy(dg, g.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(du, u.data(), n * 4, cudaMemcpyHostToDevice);
    lb::swiglu(dg, du, doo, n);
    cudaDeviceSynchronize();
    cudaMemcpy(ogot.data(), doo, n * 4, cudaMemcpyDeviceToHost);
    double mr, ma = cmp(ogot, oref, mr);
    bool p = ma < 1e-4 && mr < 1e-3;
    printf("SwiGLU[%d]: max_abs=%.2e max_rel=%.2e %s\n", n, ma, mr, p ? "PASS" : "FAIL");
    ok &= p; cudaFree(dg); cudaFree(du); cudaFree(doo);
  }

  printf("== %s ==\n", ok ? "PASS — RMSNorm + SwiGLU match CPU reference" : "FAIL");
  return ok ? 0 : 1;
}
