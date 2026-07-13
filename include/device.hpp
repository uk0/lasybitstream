// Host-side declarations for the CUDA device layer (compiled by nvcc + clang++).
// No CUDA types leak here so the C++ TUs (clang++) can include it freely.
#pragma once
#include <cstdint>
#include <cstddef>

namespace lb {

struct DeviceInfo {
  char name[256] = {0};
  int cc_major = 0, cc_minor = 0;
  int sm_count = 0;
  double mem_gb = 0;
  bool ok = false;
};

DeviceInfo device_info();

// Upload `n` bytes to the GB10 and reduce a byte checksum on-device. Proves the
// clang(host)+nvcc(device) sm_121 path works end-to-end on real model weights,
// and times the H2D copy (a proxy for the unified-memory bandwidth path).
uint64_t device_checksum(const uint8_t* host, size_t n,
                         double* ms_upload, double* ms_kernel, double* gbps);

}  // namespace lb
