#include "device.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace lb {

#define CU(call)                                                          \
  do {                                                                    \
    cudaError_t e_ = (call);                                              \
    if (e_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), \
              __FILE__, __LINE__);                                        \
      return {};                                                          \
    }                                                                     \
  } while (0)

DeviceInfo device_info() {
  DeviceInfo d;
  int dev = 0;
  cudaDeviceProp p{};
  if (cudaGetDevice(&dev) != cudaSuccess) return d;
  if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return d;
  snprintf(d.name, sizeof(d.name), "%s", p.name);
  d.cc_major = p.major;
  d.cc_minor = p.minor;
  d.sm_count = p.multiProcessorCount;
  d.mem_gb = p.totalGlobalMem / 1e9;
  d.ok = true;
  return d;
}

// Grid-stride byte sum into a 64-bit accumulator.
__global__ void sum_bytes(const uint8_t* __restrict__ d, size_t n,
                          unsigned long long* __restrict__ out) {
  unsigned long long local = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    local += d[i];
  atomicAdd(out, local);
}

uint64_t device_checksum(const uint8_t* host, size_t n, double* ms_upload,
                         double* ms_kernel, double* gbps) {
  uint8_t* d = nullptr;
  unsigned long long* acc = nullptr;
  cudaEvent_t t0, t1, t2;
  cudaError_t e;
  if ((e = cudaMalloc(&d, n)) != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(%zu) failed: %s\n", n, cudaGetErrorString(e));
    return 0;
  }
  cudaMalloc(&acc, sizeof(unsigned long long));
  cudaMemset(acc, 0, sizeof(unsigned long long));
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventCreate(&t2);

  cudaEventRecord(t0);
  cudaMemcpy(d, host, n, cudaMemcpyHostToDevice);
  cudaEventRecord(t1);

  int block = 256;
  int grid = 1024;
  sum_bytes<<<grid, block>>>(d, n, acc);
  cudaEventRecord(t2);
  cudaEventSynchronize(t2);

  float up_ms = 0, k_ms = 0;
  cudaEventElapsedTime(&up_ms, t0, t1);
  cudaEventElapsedTime(&k_ms, t1, t2);
  if (ms_upload) *ms_upload = up_ms;
  if (ms_kernel) *ms_kernel = k_ms;
  if (gbps) *gbps = (n / 1e9) / (up_ms / 1e3);

  unsigned long long out = 0;
  cudaMemcpy(&out, acc, sizeof(out), cudaMemcpyDeviceToHost);

  cudaFree(d);
  cudaFree(acc);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaEventDestroy(t2);
  return out;
}

}  // namespace lb
