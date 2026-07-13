#include "vision_tower.hpp"
#include "vision.hpp"
#include "forward.hpp"     // gemm_bf16, add_inplace
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cmath>

namespace lb {

static const int H = 1152, NHEADS = 16, HD = 72, INTER = 4304, DEPTH = 27;
static const int PATCH_IN = 1536, MERGE = 2, MOUT = 5120, MERGE_H = H * MERGE * MERGE;
static const int GRID_SIDE = 48;
static const float LNEPS = 1e-6f, RTHETA = 10000.f;

static float* dalloc(int64_t n) { float* p; cudaMalloc(&p, n * sizeof(float)); return p; }

void* VisionTower::raw(const std::string& n) {
  auto it = cache_.find(n); if (it != cache_.end()) return it->second;
  const auto& ti = st_.info(n); void* d; cudaMalloc(&d, ti.nbytes());
  cudaMemcpy(d, st_.data(n), ti.nbytes(), cudaMemcpyHostToDevice); cache_[n] = d; return d;
}
float* VisionTower::f32(const std::string& n) {
  std::string k = n + "#f"; auto it = cache_.find(k); if (it != cache_.end()) return (float*)it->second;
  const auto& ti = st_.info(n); int64_t ne = ti.numel();
  const uint16_t* s = (const uint16_t*)st_.data(n); std::vector<float> h(ne);
  for (int64_t i = 0; i < ne; ++i) { uint32_t b = (uint32_t)s[i] << 16; __builtin_memcpy(&h[i], &b, 4); }
  float* d = dalloc(ne); cudaMemcpy(d, h.data(), ne * 4, cudaMemcpyHostToDevice); cache_[k] = d; return d;
}

void VisionTower::load(const std::string& model_dir) { st_.open(model_dir + "/model.safetensors"); }

int VisionTower::encode(const float* d_pix, int gt, int gh, int gw, float* out) {
  std::string V = "model.visual.";
  int S = gt * gh * gw, MERGED = gt * (gh / MERGE) * (gw / MERGE);
  auto lin_ = [&](const float* x, const std::string& pfx, float* y, int M, int o, int in) {
    gemm_bf16(x, (const uint16_t*)raw(pfx + ".weight"), y, M, o, in);
    add_bias(y, f32(pfx + ".bias"), M, o);
  };

  // host: bilinear pos_embed (block order) + 2D RoPE cos/sin
  std::vector<float> peh((int64_t)GRID_SIDE * GRID_SIDE * H);
  { const uint16_t* s = (const uint16_t*)st_.data(V + "pos_embed.weight");
    for (size_t i = 0; i < peh.size(); ++i) { uint32_t b = (uint32_t)s[i] << 16; __builtin_memcpy(&peh[i], &b, 4); } }
  auto lin = [](int n, int i) { return n == 1 ? 0.f : (float)i * (GRID_SIDE - 1) / (n - 1); };
  std::vector<float> pos((int64_t)S * H), cosv((int64_t)S * HD), sinv((int64_t)S * HD);
  int mw = gw / MERGE;
  float inv_freq[HD / 4];
  for (int fi = 0; fi < HD / 4; ++fi) inv_freq[fi] = 1.f / powf(RTHETA, (float)(2 * fi) / (HD / 2));
  for (int p = 0; p < S; ++p) {
    int jj = p % MERGE, ii = (p / MERGE) % MERGE, bj = (p / (MERGE * MERGE)) % mw, bi = (p / (MERGE * MERGE)) / mw;
    int r = bi * MERGE + ii, c = bj * MERGE + jj;
    float hy = lin(gh, r), wx = lin(gw, c);
    int hf = (int)hy, wf = (int)wx, hc = hf + 1 < GRID_SIDE ? hf + 1 : GRID_SIDE - 1, wc = wf + 1 < GRID_SIDE ? wf + 1 : GRID_SIDE - 1;
    float dh = hy - hf, dw = wx - wf;
    float wt[4] = {(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw};
    int id[4] = {hf * GRID_SIDE + wf, hf * GRID_SIDE + wc, hc * GRID_SIDE + wf, hc * GRID_SIDE + wc};
    for (int d = 0; d < H; ++d)
      pos[(int64_t)p * H + d] = wt[0] * peh[(int64_t)id[0] * H + d] + wt[1] * peh[(int64_t)id[1] * H + d] +
                                wt[2] * peh[(int64_t)id[2] * H + d] + wt[3] * peh[(int64_t)id[3] * H + d];
    for (int fi = 0; fi < HD / 4; ++fi) {
      float er = r * inv_freq[fi], ec = c * inv_freq[fi];
      cosv[(int64_t)p * HD + fi] = cosf(er); cosv[(int64_t)p * HD + HD / 4 + fi] = cosf(ec);
      cosv[(int64_t)p * HD + HD / 2 + fi] = cosf(er); cosv[(int64_t)p * HD + HD / 2 + HD / 4 + fi] = cosf(ec);
      sinv[(int64_t)p * HD + fi] = sinf(er); sinv[(int64_t)p * HD + HD / 4 + fi] = sinf(ec);
      sinv[(int64_t)p * HD + HD / 2 + fi] = sinf(er); sinv[(int64_t)p * HD + HD / 2 + HD / 4 + fi] = sinf(ec);
    }
  }
  float* d_pos = dalloc((int64_t)S * H); cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice);
  float* d_cos = dalloc((int64_t)S * HD); cudaMemcpy(d_cos, cosv.data(), cosv.size() * 4, cudaMemcpyHostToDevice);
  float* d_sin = dalloc((int64_t)S * HD); cudaMemcpy(d_sin, sinv.data(), sinv.size() * 4, cudaMemcpyHostToDevice);

  float* h = dalloc((int64_t)S * H);
  float *ln = dalloc((int64_t)S * H), *qkv = dalloc((int64_t)S * 3 * H), *att = dalloc((int64_t)S * H);
  float *proj = dalloc((int64_t)S * H), *fc1 = dalloc((int64_t)S * INTER), *fc2 = dalloc((int64_t)S * H);
  float *q = dalloc((int64_t)S * H), *k = dalloc((int64_t)S * H), *vv = dalloc((int64_t)S * H);

  lin_(d_pix, V + "patch_embed.proj", h, S, H, PATCH_IN);
  add_inplace(h, d_pos, S * H);
  for (int L = 0; L < DEPTH; ++L) {
    std::string b = V + "blocks." + std::to_string(L) + ".";
    layernorm(h, f32(b + "norm1.weight"), f32(b + "norm1.bias"), LNEPS, ln, S, H);
    lin_(ln, b + "attn.qkv", qkv, S, 3 * H, H);
    cudaMemcpy2D(q, H * 4, qkv, 3 * H * 4, H * 4, S, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(k, H * 4, qkv + H, 3 * H * 4, H * 4, S, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(vv, H * 4, qkv + 2 * H, 3 * H * 4, H * 4, S, cudaMemcpyDeviceToDevice);
    rope_vision(q, k, d_cos, d_sin, S, NHEADS, HD);
    vision_attention(q, k, vv, att, S, NHEADS, HD);
    lin_(att, b + "attn.proj", proj, S, H, H);
    add_inplace(h, proj, S * H);
    layernorm(h, f32(b + "norm2.weight"), f32(b + "norm2.bias"), LNEPS, ln, S, H);
    lin_(ln, b + "mlp.linear_fc1", fc1, S, INTER, H);
    gelu_tanh(fc1, fc1, (int64_t)S * INTER);
    lin_(fc1, b + "mlp.linear_fc2", fc2, S, H, INTER);
    add_inplace(h, fc2, S * H);
  }
  float* mln = dalloc((int64_t)S * H);
  layernorm(h, f32(V + "merger.norm.weight"), f32(V + "merger.norm.bias"), LNEPS, mln, S, H);
  float* m1 = dalloc((int64_t)MERGED * MERGE_H);
  lin_(mln, V + "merger.linear_fc1", m1, MERGED, MERGE_H, MERGE_H);
  gelu_erf(m1, m1, (int64_t)MERGED * MERGE_H);
  lin_(m1, V + "merger.linear_fc2", out, MERGED, MOUT, MERGE_H);
  cudaDeviceSynchronize();
  cudaFree(d_pos); cudaFree(d_cos); cudaFree(d_sin); cudaFree(h); cudaFree(ln); cudaFree(qkv);
  cudaFree(att); cudaFree(proj); cudaFree(fc1); cudaFree(fc2); cudaFree(q); cudaFree(k); cudaFree(vv);
  cudaFree(mln); cudaFree(m1);
  return MERGED;
}

}  // namespace lb
