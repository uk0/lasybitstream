// Verify the native Qwen3.6-VL vision tower against the golden reference
// (scripts/vis_ref.py -> test/vis_{pixels,grid,emb}). Loads model.visual.*,
// runs patch-embed -> +pos-embed -> 27 ViT blocks (2D RoPE) -> merger, and
// compares the merged image embeddings.
//   ./lbtest_vis <model_dir> [test_dir]
#include "safetensors.hpp"
#include "vision.hpp"
#include "forward.hpp"     // gemm_bf16
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <cmath>

using namespace lb;

static const int H = 1152, NHEADS = 16, HD = 72, INTER = 4304, DEPTH = 27;
static const int PATCH_IN = 1536, MERGE = 2, MOUT = 5120, MERGE_H = H * MERGE * MERGE;  // 4608
static const int GRID_SIDE = 48;                        // sqrt(num_position_embeddings=2304)
static const float LNEPS = 1e-6f, RTHETA = 10000.f;

static float* dalloc(int64_t n) { float* p; cudaMalloc(&p, n * sizeof(float)); return p; }

struct VW {  // vision weight loader
  SafeTensors st;
  std::map<std::string, void*> cache;
  void* raw(const std::string& n) {
    auto it = cache.find(n); if (it != cache.end()) return it->second;
    const auto& ti = st.info(n); void* d; cudaMalloc(&d, ti.nbytes());
    cudaMemcpy(d, st.data(n), ti.nbytes(), cudaMemcpyHostToDevice); cache[n] = d; return d;
  }
  float* f32(const std::string& n) {
    std::string k = n + "#f"; auto it = cache.find(k); if (it != cache.end()) return (float*)it->second;
    const auto& ti = st.info(n); int64_t ne = ti.numel();
    const uint16_t* s = (const uint16_t*)st.data(n); std::vector<float> h(ne);
    for (int64_t i = 0; i < ne; ++i) { uint32_t b = (uint32_t)s[i] << 16; __builtin_memcpy(&h[i], &b, 4); }
    float* d = dalloc(ne); cudaMemcpy(d, h.data(), ne * 4, cudaMemcpyHostToDevice); cache[k] = d; return d;
  }
  const uint16_t* bf(const std::string& n) { return (const uint16_t*)raw(n); }
};

// y[M,out] = x[M,in] @ W_bf16[out,in]^T + bias   (bias f32)
static void linear(VW& w, const float* x, const std::string& pfx, float* y, int M, int out, int in) {
  gemm_bf16(x, w.bf(pfx + ".weight"), y, M, out, in);
  add_bias(y, w.f32(pfx + ".bias"), M, out);
}

int main(int argc, char** argv) {
  std::string mdir = argc > 1 ? argv[1] : "/model";
  std::string tdir = argc > 2 ? argv[2] : "test";
  VW w; w.st.open(mdir + "/model.safetensors");
  std::string V = "model.visual.";

  // grid + pixels
  int gt, gh, gw;
  { FILE* f = fopen((tdir + "/vis_grid.i32").c_str(), "rb"); if (!f) { printf("need vis_grid.i32\n"); return 2; }
    int g[3]; if (fread(g, 4, 3, f) != 3) return 2; fclose(f); gt = g[0]; gh = g[1]; gw = g[2]; }
  int S = gt * gh * gw, MERGED = gt * (gh / MERGE) * (gw / MERGE);
  printf("grid t=%d h=%d w=%d  S=%d merged=%d\n", gt, gh, gw, S, MERGED);
  std::vector<float> pix((int64_t)S * PATCH_IN);
  { FILE* f = fopen((tdir + "/vis_pixels.f32").c_str(), "rb"); fread(pix.data(), 4, pix.size(), f); fclose(f); }
  float* d_pix = dalloc((int64_t)S * PATCH_IN);
  cudaMemcpy(d_pix, pix.data(), pix.size() * 4, cudaMemcpyHostToDevice);

  // ---- host: interpolated pos_embed (block order) + 2D RoPE cos/sin ----
  const uint16_t* pe = w.bf(V + "pos_embed.weight");           // device; read host copy
  std::vector<float> peh((int64_t)GRID_SIDE * GRID_SIDE * H);
  { const uint16_t* s = (const uint16_t*)w.st.data(V + "pos_embed.weight");
    for (size_t i = 0; i < peh.size(); ++i) { uint32_t b = (uint32_t)s[i] << 16; __builtin_memcpy(&peh[i], &b, 4); } }
  (void)pe;
  auto lin = [](int n, int i) { return n == 1 ? 0.f : (float)i * (GRID_SIDE - 1) / (n - 1); };
  std::vector<float> pos((int64_t)S * H), cosv((int64_t)S * HD), sinv((int64_t)S * HD);
  int mh = gh / MERGE, mw = gw / MERGE;
  float inv_freq[HD / 4];                                       // dim=36 -> 18 freqs
  for (int fi = 0; fi < HD / 4; ++fi) inv_freq[fi] = 1.f / powf(RTHETA, (float)(2 * fi) / (HD / 2));
  for (int p = 0; p < S; ++p) {
    int jj = p % MERGE, ii = (p / MERGE) % MERGE, bj = (p / (MERGE * MERGE)) % mw, bi = (p / (MERGE * MERGE)) / mw;
    int r = bi * MERGE + ii, c = bj * MERGE + jj;              // full-res patch coords
    // bilinear pos-embed
    float hy = lin(gh, r), wx = lin(gw, c);
    int hf = (int)hy, wf = (int)wx, hc = hf + 1 < GRID_SIDE ? hf + 1 : GRID_SIDE - 1, wc = wf + 1 < GRID_SIDE ? wf + 1 : GRID_SIDE - 1;
    float dh = hy - hf, dw = wx - wf;
    float wt[4] = {(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw};
    int id[4] = {hf * GRID_SIDE + wf, hf * GRID_SIDE + wc, hc * GRID_SIDE + wf, hc * GRID_SIDE + wc};
    for (int d = 0; d < H; ++d)
      pos[(int64_t)p * H + d] = wt[0] * peh[(int64_t)id[0] * H + d] + wt[1] * peh[(int64_t)id[1] * H + d] +
                                wt[2] * peh[(int64_t)id[2] * H + d] + wt[3] * peh[(int64_t)id[3] * H + d];
    // 2D rope: emb36 = [r*inv_freq | c*inv_freq]; emb72 = cat(emb36, emb36)
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

  // ---- tower ----
  float* h = dalloc((int64_t)S * H);
  float *ln = dalloc((int64_t)S * H), *qkv = dalloc((int64_t)S * 3 * H), *att = dalloc((int64_t)S * H);
  float *proj = dalloc((int64_t)S * H), *fc1 = dalloc((int64_t)S * INTER), *fc2 = dalloc((int64_t)S * H);
  float *q = dalloc((int64_t)S * H), *k = dalloc((int64_t)S * H), *vv = dalloc((int64_t)S * H);

  auto cmp = [&](const char* nm, const float* dev, int64_t n) {
    FILE* f = fopen((tdir + "/" + nm).c_str(), "rb"); if (!f) { printf("  (no %s)\n", nm); return; }
    std::vector<float> r(n), g(n); fread(r.data(), 4, n, f); fclose(f);
    cudaMemcpy(g.data(), dev, n * 4, cudaMemcpyDeviceToHost);
    double ma = 0, mr = 0;
    for (int64_t i = 0; i < n; ++i) { double d = std::fabs((double)g[i] - r[i]); ma = std::max(ma, d); mr = std::max(mr, d / (std::fabs((double)r[i]) + 1e-2)); }
    printf("  %-16s max_abs=%.2e max_rel=%.2e\n", nm, ma, mr);
  };

  linear(w, d_pix, V + "patch_embed.proj", h, S, H, PATCH_IN);   // patch embed
  add_inplace(h, d_pos, S * H);                                  // + pos embed
  cudaDeviceSynchronize(); cmp("vis_h_in.f32", h, (int64_t)S * H);

  for (int L = 0; L < DEPTH; ++L) {
    std::string b = V + "blocks." + std::to_string(L) + ".";
    layernorm(h, w.f32(b + "norm1.weight"), w.f32(b + "norm1.bias"), LNEPS, ln, S, H);
    linear(w, ln, b + "attn.qkv", qkv, S, 3 * H, H);
    // split [S, 3H] -> q,k,v [S,H]  (layout: 3*NHEADS*HD, first H=q, next H=k, next H=v)
    cudaMemcpy2D(q, H * 4, qkv, 3 * H * 4, H * 4, S, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(k, H * 4, qkv + H, 3 * H * 4, H * 4, S, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(vv, H * 4, qkv + 2 * H, 3 * H * 4, H * 4, S, cudaMemcpyDeviceToDevice);
    rope_vision(q, k, d_cos, d_sin, S, NHEADS, HD);
    vision_attention(q, k, vv, att, S, NHEADS, HD);
    linear(w, att, b + "attn.proj", proj, S, H, H);
    add_inplace(h, proj, S * H);
    if (L == 0) { cudaDeviceSynchronize(); cmp("vis_h_postattn.f32", h, (int64_t)S * H); }
    layernorm(h, w.f32(b + "norm2.weight"), w.f32(b + "norm2.bias"), LNEPS, ln, S, H);
    linear(w, ln, b + "mlp.linear_fc1", fc1, S, INTER, H);
    gelu_tanh(fc1, fc1, (int64_t)S * INTER);
    linear(w, fc1, b + "mlp.linear_fc2", fc2, S, H, INTER);
    add_inplace(h, fc2, S * H);
    if (L == 0) { cudaDeviceSynchronize(); cmp("vis_h_b0.f32", h, (int64_t)S * H); }
  }
  cudaDeviceSynchronize(); cmp("vis_h_pre.f32", h, (int64_t)S * H);

  // merger: LayerNorm(1152) -> view[MERGED,4608] -> fc1 -> gelu(erf) -> fc2 -> [MERGED,5120]
  float* mln = dalloc((int64_t)S * H);
  layernorm(h, w.f32(V + "merger.norm.weight"), w.f32(V + "merger.norm.bias"), LNEPS, mln, S, H);
  float* m1 = dalloc((int64_t)MERGED * MERGE_H);
  linear(w, mln, V + "merger.linear_fc1", m1, MERGED, MERGE_H, MERGE_H);  // view is implicit (contiguous)
  gelu_erf(m1, m1, (int64_t)MERGED * MERGE_H);
  float* emb = dalloc((int64_t)MERGED * MOUT);
  linear(w, m1, V + "merger.linear_fc2", emb, MERGED, MOUT, MERGE_H);
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }

  // compare
  std::vector<float> got((int64_t)MERGED * MOUT), ref((int64_t)MERGED * MOUT);
  cudaMemcpy(got.data(), emb, got.size() * 4, cudaMemcpyDeviceToHost);
  { FILE* f = fopen((tdir + "/vis_emb.f32").c_str(), "rb"); if (!f) { printf("need vis_emb.f32\n"); return 2; }
    fread(ref.data(), 4, ref.size(), f); fclose(f); }
  double maxabs = 0, maxrel = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    double d = std::fabs((double)got[i] - ref[i]); maxabs = std::max(maxabs, d);
    maxrel = std::max(maxrel, d / (std::fabs((double)ref[i]) + 1e-2));
  }
  bool ok = maxrel < 5e-2 && maxabs < 5e-2;
  printf("vision emb[%d,%d]: max_abs=%.2e max_rel=%.2e %s\n", MERGED, MOUT, maxabs, maxrel, ok ? "PASS" : "FAIL");
  printf("== %s ==\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
