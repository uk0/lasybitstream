// Verify the NVFP4 GEMM (y = x @ dequant(W)^T) against a CPU reference that uses
// the already-verified dequant weight (test/ref_gate0.f32) and the same inputs.
#include "safetensors.hpp"
#include "nvfp4.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : "/home/neo/models/Qwen3.6-27B-NVFP4";
  const char* ref = argc > 2 ? argv[2] : "test/ref_gate0.f32";
  const int M = 4, OUT = 256, G = 16;
  const std::string P = "model.language_model.layers.0.mlp.gate_proj.";

  lb::SafeTensors st;
  st.open(std::string(dir) + "/model.safetensors");
  const lb::TensorInfo& pk = st.info(P + "weight_packed");
  int64_t IN = pk.shape[1] * 2, in2 = IN / 2, gg = IN / G;
  float gscale;
  std::memcpy(&gscale, st.data(P + "weight_global_scale"), 4);

  // Reference dequant weight W[:256] (produced by scripts/nvfp4_ref.py).
  std::vector<float> refW((size_t)OUT * IN);
  std::ifstream rf(ref, std::ios::binary);
  if (!rf) { printf("cannot open reference %s\n", ref); return 2; }
  rf.read((char*)refW.data(), (std::streamsize)refW.size() * 4);

  // Deterministic activations x[M, IN].
  std::vector<float> x((size_t)M * IN);
  for (int m = 0; m < M; ++m)
    for (int i = 0; i < IN; ++i)
      x[(size_t)m * IN + i] = std::sin(0.001f * (i + 1) + 0.5f * m) * 0.7f;

  // CPU reference: y_ref = x @ refW^T (double accumulate).
  std::vector<float> yref((size_t)M * OUT);
  for (int m = 0; m < M; ++m)
    for (int o = 0; o < OUT; ++o) {
      double a = 0;
      const float* xr = &x[(size_t)m * IN];
      const float* wr = &refW[(size_t)o * IN];
      for (int i = 0; i < IN; ++i) a += (double)xr[i] * wr[i];
      yref[(size_t)m * OUT + o] = (float)a;
    }

  // Device GEMM (weight dequanted on the fly from packed).
  float *dx = nullptr, *dy = nullptr;
  uint8_t *dp = nullptr, *ds = nullptr;
  cudaMalloc(&dx, (size_t)M * IN * 4);
  cudaMalloc(&dy, (size_t)M * OUT * 4);
  cudaMalloc(&dp, (size_t)OUT * in2);
  cudaMalloc(&ds, (size_t)OUT * gg);
  cudaMemcpy(dx, x.data(), (size_t)M * IN * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dp, st.data(P + "weight_packed"), (size_t)OUT * in2, cudaMemcpyHostToDevice);
  cudaMemcpy(ds, st.data(P + "weight_scale"), (size_t)OUT * gg, cudaMemcpyHostToDevice);
  lb::gemm_nvfp4(dx, dp, ds, gscale, dy, M, OUT, IN, G);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
  std::vector<float> ygot((size_t)M * OUT);
  cudaMemcpy(ygot.data(), dy, (size_t)M * OUT * 4, cudaMemcpyDeviceToHost);

  double maxabs = 0, maxrel = 0, sum = 0;
  for (size_t i = 0; i < ygot.size(); ++i) {
    double d = std::fabs((double)ygot[i] - yref[i]);
    maxabs = std::max(maxabs, d);
    sum += d;
    maxrel = std::max(maxrel, d / (std::fabs((double)yref[i]) + 1e-6));
  }
  printf("GEMM y[%d,%d] vs CPU ref: max_abs=%.2e mean_abs=%.2e max_rel=%.2e  (ref[0,0]=%.5f got[0,0]=%.5f)\n",
         M, OUT, maxabs, sum / ygot.size(), maxrel, yref[0], ygot[0]);
  bool pass = maxrel < 1e-3 && maxabs < 1e-2;
  printf("== %s ==\n", pass ? "PASS — NVFP4 GEMM matches dequant+matmul reference" : "FAIL");
  cudaFree(dx); cudaFree(dy); cudaFree(dp); cudaFree(ds);
  return pass ? 0 : 1;
}
