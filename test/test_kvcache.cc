// Verify the NVMe+RAM hybrid KV cache: append more positions than fit in the GPU hot
// window (forcing spill to the NVMe mmap), then read every block back and check integrity.
//   ./lbtest_kv [nvme_path]
#include "kvcache.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

using namespace lb;

int main(int argc, char** argv) {
  const char* nvme = argc > 1 ? argv[1] : "/tmp/lb_kv.bin";
  const int BT = 8, KD = 16, HOT = 4, N = 200;          // 200 pos / 8 = 25 blocks; only 4 hot
  HybridKVCache kv; kv.init(BT, KD, HOT, 512, nvme);
  float* d; cudaMalloc(&d, KD * 4);
  std::vector<float> h(KD);
  for (int p = 0; p < N; ++p) {
    for (int i = 0; i < KD; ++i) h[i] = p * 100.0f + i;  // known pattern per (pos, dim)
    cudaMemcpy(d, h.data(), KD * 4, cudaMemcpyHostToDevice);
    kv.append(d);
  }
  printf("appended %d positions -> %d blocks (%d hot in GPU, %d spilled to NVMe)\n",
         kv.positions(), kv.num_blocks(), kv.hot_resident(), kv.spilled_blocks());

  int bad = 0;
  std::vector<float> blk((size_t)BT * KD);
  for (int b = 0; b < kv.num_blocks(); ++b) {
    const float* dev = kv.block_dev(b);                  // hot slot or staged from NVMe
    cudaMemcpy(blk.data(), dev, (size_t)BT * KD * 4, cudaMemcpyDeviceToHost);
    for (int t = 0; t < BT; ++t) {
      int p = b * BT + t; if (p >= N) break;
      for (int i = 0; i < KD; ++i) {
        float exp = p * 100.0f + i;
        if (std::fabs(blk[t * KD + i] - exp) > 1e-3) { if (bad < 5) printf("  mismatch p=%d i=%d got=%.1f exp=%.1f\n", p, i, blk[t * KD + i], exp); ++bad; }
      }
    }
  }
  cudaFree(d);
  bool ok = bad == 0 && kv.spilled_blocks() > 0;
  printf("== %s == (%d mismatches, %d blocks spilled+reloaded from NVMe)\n",
         ok ? "PASS — hybrid KV spill+reload integrity" : "FAIL", bad, kv.spilled_blocks());
  return ok ? 0 : 1;
}
