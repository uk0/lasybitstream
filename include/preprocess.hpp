// Image preprocessing for the vision path: base64 decode + image (JPEG/PNG/…) decode
// (vendored stb_image) + Qwen-style smart-resize/normalize/patchify into pixel_values
// [t*h*w, 1536] plus the grid (t,h,w), ready for VisionTower::encode.
#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace lb {

struct PreImage {
  std::vector<float> pixels;   // [t*h*w, 1536]
  int t = 1, h = 0, w = 0;     // grid in patch units
  bool ok = false;
};

std::vector<uint8_t> base64_decode(const std::string& s);

// Decode encoded image bytes and preprocess. Normalization mean=std=0.5 (rescale 1/255),
// patch 16 / temporal 2 / merge 2; patches laid out in 2x2-block order (as the tower expects).
PreImage preprocess_image(const uint8_t* bytes, size_t n);

}  // namespace lb
