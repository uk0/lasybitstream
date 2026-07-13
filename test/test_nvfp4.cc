// Verify the CUDA NVFP4 dequant kernel against the canonical reference
// (test/ref_gate0.f32, produced by scripts/nvfp4_ref.py via compressed_tensors).
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
  const int64_t ROWS = 256, G = 16;
  const std::string P = "model.language_model.layers.0.mlp.gate_proj.";

  lb::SafeTensors st;
  st.open(std::string(dir) + "/model.safetensors");
  const lb::TensorInfo& pk = st.info(P + "weight_packed");  // U8 [OUT, IN/2]
  int64_t IN = pk.shape[1] * 2, in2 = IN / 2, gg = IN / G;
  float gscale;
  std::memcpy(&gscale, st.data(P + "weight_global_scale"), 4);
  printf("layer0.mlp.gate_proj: OUT=%ld IN=%ld group=%ld gscale=%.1f (testing first %ld rows)\n",
         (long)pk.shape[0], (long)IN, (long)G, gscale, (long)ROWS);

  size_t pbytes = (size_t)ROWS * in2, sbytes = (size_t)ROWS * gg, obytes = (size_t)ROWS * IN * 4;
  uint8_t *dp = nullptr, *ds = nullptr;
  float* dout = nullptr;
  cudaMalloc(&dp, pbytes);
  cudaMalloc(&ds, sbytes);
  cudaMalloc(&dout, obytes);
  cudaMemcpy(dp, st.data(P + "weight_packed"), pbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(ds, st.data(P + "weight_scale"), sbytes, cudaMemcpyHostToDevice);

  lb::dequant_nvfp4(dp, ds, gscale, dout, ROWS, IN, (int)G);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }

  std::vector<float> got((size_t)ROWS * IN), refv((size_t)ROWS * IN);
  cudaMemcpy(got.data(), dout, obytes, cudaMemcpyDeviceToHost);
  std::ifstream rf(ref, std::ios::binary);
  if (!rf) { printf("cannot open reference %s\n", ref); return 2; }
  rf.read((char*)refv.data(), (std::streamsize)refv.size() * 4);

  double maxabs = 0, sum = 0, maxrel = 0;
  int nbad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    double d = std::fabs((double)got[i] - refv[i]);
    maxabs = std::max(maxabs, d);
    sum += d;
    maxrel = std::max(maxrel, d / (std::fabs((double)refv[i]) + 1e-8));
    if (d > 1e-4) nbad++;
  }
  printf("dequant vs reference: %zu elems  max_abs=%.2e  mean_abs=%.2e  max_rel=%.2e  bad(>1e-4)=%d\n",
         got.size(), maxabs, sum / got.size(), maxrel, nbad);
  bool pass = maxabs < 1e-3 && nbad == 0;
  printf("== %s ==\n", pass ? "PASS — CUDA NVFP4 dequant matches canonical reference"
                            : "FAIL");
  cudaFree(dp); cudaFree(ds); cudaFree(dout);
  return pass ? 0 : 1;
}
