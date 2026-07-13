// NVFP4 (compressed-tensors nvfp4-pack-quantized) dequant on the GB10.
#pragma once
#include <cstdint>

namespace lb {

// Dequant an NVFP4 weight to f32 on the device:
//   w[o,i] = e2m1(nibble) * fp8_e4m3(weight_scale[o, i/group]) / global_scale
// Convention (verified vs compressed_tensors): low nibble = even i.
//   packed     : device U8  [rows, in/2]  (two fp4 per byte)
//   scale_fp8  : device raw fp8_e4m3 bytes [rows, in/group]
//   out        : device f32 [rows, in]
void dequant_nvfp4(const uint8_t* packed, const uint8_t* scale_fp8, float global_scale,
                   float* out, int64_t rows, int64_t in, int group);

}  // namespace lb
