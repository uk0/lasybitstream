// The Qwen3.6-VL vision module: loads model.visual.* (BF16) and turns preprocessed
// image patches into LLM-space image embeddings [merged, 5120] via patch-embed ->
// bilinear pos-embed -> 27 ViT blocks (2D RoPE) -> 2x2 patch merger (the proj to
// out_hidden_size). Verified against scripts/vis_ref.py.
#pragma once
#include "safetensors.hpp"
#include <string>

namespace lb {

class VisionTower {
 public:
  ~VisionTower();                                     // frees uploaded device weights
  void load(const std::string& model_dir);           // opens model.safetensors, mmap
  // Encode one image: `pixels_dev` [S,1536] (S = t*h*w patches, device f32),
  // grid (t,h,w). Writes merged embeddings [merged, 5120] to out_dev (device,
  // caller-allocated to at least merged*5120). Returns merged token count.
  int encode(const float* pixels_dev, int t, int h, int w, float* out_dev);
  int out_hidden() const { return 5120; }

 private:
  SafeTensors st_;
  std::map<std::string, void*> cache_;
  void* raw(const std::string& n);
  float* f32(const std::string& n);
};

}  // namespace lb
