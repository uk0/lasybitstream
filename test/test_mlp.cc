// Verify the composed MLP block  y = down(silu(gate(x)) * up(x))  built from the
// verified NVFP4 GEMM + SwiGLU kernels, against a numpy reference on real weights.
#include "safetensors.hpp"
#include "nvfp4.hpp"
#include "ops.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : "/home/neo/models/Qwen3.6-27B-NVFP4";
  const int M = 2, HID = 5120, INTER = 17408, G = 16;
  const std::string L = "model.language_model.layers.0.mlp.";

  lb::SafeTensors st;
  st.open(std::string(dir) + "/model.safetensors");

  struct QW { uint8_t *p = nullptr, *s = nullptr; float g = 0; int64_t out = 0, in = 0; };
  auto up = [&](const std::string& pfx) {
    QW w;
    const lb::TensorInfo& pk = st.info(pfx + "weight_packed");
    w.out = pk.shape[0]; w.in = pk.shape[1] * 2;
    std::memcpy(&w.g, st.data(pfx + "weight_global_scale"), 4);
    size_t pb = (size_t)w.out * pk.shape[1], sb = (size_t)w.out * (w.in / G);
    cudaMalloc(&w.p, pb); cudaMalloc(&w.s, sb);
    cudaMemcpy(w.p, st.data(pfx + "weight_packed"), pb, cudaMemcpyHostToDevice);
    cudaMemcpy(w.s, st.data(pfx + "weight_scale"), sb, cudaMemcpyHostToDevice);
    return w;
  };
  QW gate = up(L + "gate_proj."), upw = up(L + "up_proj."), down = up(L + "down_proj.");

  std::vector<float> x((size_t)M * HID);
  std::ifstream xf("test/mlp_x.f32", std::ios::binary);
  if (!xf) { printf("run scripts/mlp_ref.py first (missing test/mlp_x.f32)\n"); return 2; }
  xf.read((char*)x.data(), (std::streamsize)x.size() * 4);

  float *dx, *dgate, *dup, *dh, *dy;
  cudaMalloc(&dx, (size_t)M * HID * 4);
  cudaMalloc(&dgate, (size_t)M * INTER * 4);
  cudaMalloc(&dup, (size_t)M * INTER * 4);
  cudaMalloc(&dh, (size_t)M * INTER * 4);
  cudaMalloc(&dy, (size_t)M * HID * 4);
  cudaMemcpy(dx, x.data(), (size_t)M * HID * 4, cudaMemcpyHostToDevice);

  lb::gemm_nvfp4(dx, gate.p, gate.s, gate.g, dgate, M, gate.out, gate.in, G);  // [M, INTER]
  lb::gemm_nvfp4(dx, upw.p, upw.s, upw.g, dup, M, upw.out, upw.in, G);
  lb::swiglu(dgate, dup, dh, (int64_t)M * INTER);
  lb::gemm_nvfp4(dh, down.p, down.s, down.g, dy, M, down.out, down.in, G);     // [M, HID]
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }

  std::vector<float> ygot((size_t)M * HID), yref((size_t)M * HID);
  cudaMemcpy(ygot.data(), dy, (size_t)M * HID * 4, cudaMemcpyDeviceToHost);
  std::ifstream rf("test/mlp_yref.f32", std::ios::binary);
  rf.read((char*)yref.data(), (std::streamsize)yref.size() * 4);

  double maxabs = 0, maxrel = 0, sum = 0;
  for (size_t i = 0; i < ygot.size(); ++i) {
    double d = std::fabs((double)ygot[i] - yref[i]);
    maxabs = std::max(maxabs, d); sum += d;
    maxrel = std::max(maxrel, d / (std::fabs((double)yref[i]) + 1e-4));
  }
  printf("MLP block y[%d,%d] vs numpy ref: max_abs=%.2e mean_abs=%.2e max_rel=%.2e (ref[0]=%.4f got[0]=%.4f)\n",
         M, HID, maxabs, sum / ygot.size(), maxrel, yref[0], ygot[0]);
  bool pass = maxrel < 5e-3 && maxabs < 5e-2;
  printf("== %s ==\n", pass ? "PASS — composed MLP block matches reference" : "FAIL");
  return pass ? 0 : 1;
}
