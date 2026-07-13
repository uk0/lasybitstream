#include "engine.hpp"
#include "safetensors.hpp"
#include "nvfp4.hpp"
#include "ops.hpp"
#include "gdn.hpp"
#include "forward.hpp"
#include "vision_tower.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace lb {

// ---- model dims (Qwen3.6-27B, confirmed from config.json) ----
static const int H = 5120, NL = 64, FA_INT = 4;
static const int NH = 24, NKV = 4, HD = 256;
static const int L_KH = 16, L_VH = 48, L_KD = 128, L_VD = 128, CONV = 4;
static const int ROT = 64, GROUP = 16, VOCAB = 248320;
static const float EPS = 1e-6f, THETA = 1e7f;
static const int Q_SIZE = NH * HD, KV_SIZE = NKV * HD;
static const int L_Q = L_KH * L_KD, L_V = L_VH * L_VD;   // 2048, 6144
static const int INTER = 17408;

static float* dalloc(int64_t n) { float* p; cudaMalloc(&p, n * sizeof(float)); return p; }

struct Weights {
  SafeTensors st;
  std::map<std::string, void*> cache;
  std::string LP = "model.language_model.";

  void* up_raw(const std::string& n) {
    auto it = cache.find(n);
    if (it != cache.end()) return it->second;
    const auto& ti = st.info(n);
    void* d; cudaMalloc(&d, ti.nbytes());
    cudaMemcpy(d, st.data(n), ti.nbytes(), cudaMemcpyHostToDevice);
    cache[n] = d; return d;
  }
  float* up_f32(const std::string& n) {
    std::string key = n + "#f32";
    auto it = cache.find(key);
    if (it != cache.end()) return (float*)it->second;
    const auto& ti = st.info(n);
    int64_t ne = ti.numel();
    const uint16_t* src = (const uint16_t*)st.data(n);
    std::vector<float> host(ne);
    for (int64_t i = 0; i < ne; ++i) { uint32_t b = (uint32_t)src[i] << 16; __builtin_memcpy(&host[i], &b, 4); }
    float* d = dalloc(ne);
    cudaMemcpy(d, host.data(), ne * 4, cudaMemcpyHostToDevice);
    cache[key] = d; return d;
  }
  float gsf(const std::string& n) { return *(const float*)st.data(n); }
  void nvfp4(const float* x, const std::string& pfx, float* y, int M, int64_t out, int64_t in) {
    gemm_nvfp4(x, (const uint8_t*)up_raw(pfx + ".weight_packed"), (const uint8_t*)up_raw(pfx + ".weight_scale"),
               gsf(pfx + ".weight_global_scale"), y, M, out, in, GROUP);
  }
  void bf16(const float* x, const std::string& pfx, float* y, int M, int out, int in) {
    gemm_bf16(x, (const uint16_t*)up_raw(pfx + ".weight"), y, M, out, in);
  }
};

struct Buf {
  float *h, *xn, *xn2, *mix, *state, *logits;
  float *qkv, *z, *ga, *gb, *conv, *q, *k, *v, *g, *beta, *recur, *gated;
  float *qg, *aq, *agate, *ak, *av, *aout, *agated;
  float *mg, *mu, *msw;
  int *ids_d, *pos3d_d;
  void init(int T) {
    h = dalloc((int64_t)T * H); xn = dalloc((int64_t)T * H); xn2 = dalloc((int64_t)T * H);
    mix = dalloc((int64_t)T * H); state = dalloc((int64_t)L_VH * L_VD * L_KD); logits = dalloc(VOCAB);
    qkv = dalloc((int64_t)T * (L_Q * 2 + L_V)); z = dalloc((int64_t)T * L_V);
    ga = dalloc((int64_t)T * L_VH); gb = dalloc((int64_t)T * L_VH); conv = dalloc((int64_t)T * (L_Q * 2 + L_V));
    q = dalloc((int64_t)T * L_Q); k = dalloc((int64_t)T * L_Q); v = dalloc((int64_t)T * L_V);
    g = dalloc((int64_t)T * L_VH); beta = dalloc((int64_t)T * L_VH);
    recur = dalloc((int64_t)T * L_V); gated = dalloc((int64_t)T * L_V);
    qg = dalloc((int64_t)T * NH * 2 * HD); aq = dalloc((int64_t)T * Q_SIZE); agate = dalloc((int64_t)T * Q_SIZE);
    ak = dalloc((int64_t)T * KV_SIZE); av = dalloc((int64_t)T * KV_SIZE);
    aout = dalloc((int64_t)T * Q_SIZE); agated = dalloc((int64_t)T * Q_SIZE);
    mg = dalloc((int64_t)T * INTER); mu = dalloc((int64_t)T * INTER); msw = dalloc((int64_t)T * INTER);
    cudaMalloc(&ids_d, T * sizeof(int)); cudaMalloc(&pos3d_d, 3 * T * sizeof(int));
  }
};

static double compare(const std::string& ref, const float* dev, int64_t n) {
  FILE* f = fopen(ref.c_str(), "rb");
  if (!f) return -1;
  std::vector<float> r(n); size_t got = fread(r.data(), 4, n, f); fclose(f);
  if ((int64_t)got != n) return -2;
  std::vector<float> d(n); cudaMemcpy(d.data(), dev, n * 4, cudaMemcpyDeviceToHost);
  double mr = 0;
  for (int64_t i = 0; i < n; ++i) mr = std::max(mr, std::fabs((double)d[i] - r[i]) / (std::fabs((double)r[i]) + 1e-3));
  return mr;
}

struct Engine::Impl {
  Weights w;
  Buf b;
  int max_ctx = 4096;
  static const int CC = L_Q * 2 + L_V;                 // 10240 GDN conv channels
  std::vector<float*> kc, vc, gst, cst;                // per-layer caches (NL entries)
  VisionTower vt;                                      // model.visual.* module
  float* img_embeds = nullptr;                         // [merged, H] for the current image
  int img_merged = 0;
  std::string mdir;

  void alloc_caches() {
    kc.assign(NL, nullptr); vc.assign(NL, nullptr); gst.assign(NL, nullptr); cst.assign(NL, nullptr);
    for (int L = 0; L < NL; ++L) {
      if ((L % FA_INT) == (FA_INT - 1)) {
        kc[L] = dalloc((int64_t)max_ctx * NKV * HD); vc[L] = dalloc((int64_t)max_ctx * NKV * HD);
      } else {
        gst[L] = dalloc((int64_t)L_VH * L_VD * L_KD); cst[L] = dalloc((int64_t)CC * (CONV - 1));
      }
    }
  }

  // Forward `T` new tokens starting at global position `start_pos`, using and
  // updating the K/V + GDN-state + conv-state caches. `first = (start_pos == 0)`
  // means a fresh sequence (prefill). Returns the greedy next token.
  int forward_tokens(const std::vector<int>& ids, int T, int start_pos, const std::string* tdir,
                     const int* pos3d_host = nullptr, const float* img_embeds = nullptr,
                     int img_pos = -1, int img_cnt = 0) {
    bool first = (start_pos == 0);
    std::vector<int> pos3(3 * T);
    if (pos3d_host) std::copy(pos3d_host, pos3d_host + 3 * T, pos3.begin());
    else for (int a = 0; a < 3; ++a) for (int i = 0; i < T; ++i) pos3[a * T + i] = start_pos + i;
    cudaMemcpy(b.ids_d, ids.data(), T * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(b.pos3d_d, pos3.data(), 3 * T * 4, cudaMemcpyHostToDevice);
    embed_gather((const uint16_t*)w.up_raw(w.LP + "embed_tokens.weight"), b.ids_d, b.h, T, H);
    if (img_embeds && img_cnt > 0)                       // splice vision embeddings over the pads
      cudaMemcpy(b.h + (int64_t)img_pos * H, img_embeds, (int64_t)img_cnt * H * 4, cudaMemcpyDeviceToDevice);
    if (tdir) printf("embed:  max_rel=%.2e\n", compare(*tdir + "/ref_embed.f32", b.h, (int64_t)T * H));
    float scale_gdn = 1.0f / std::sqrt((float)L_KD);
    for (int L = 0; L < NL; ++L) {
      std::string lp = w.LP + "layers." + std::to_string(L) + ".";
      rmsnorm(b.h, w.up_f32(lp + "input_layernorm.weight"), EPS, b.xn, T, H);
      bool is_attn = (L % FA_INT) == (FA_INT - 1);
      if (is_attn) {
        std::string ap = lp + "self_attn";
        w.nvfp4(b.xn, ap + ".q_proj", b.qg, T, NH * 2 * HD, H);
        w.nvfp4(b.xn, ap + ".k_proj", b.ak, T, KV_SIZE, H);
        w.nvfp4(b.xn, ap + ".v_proj", b.av, T, KV_SIZE, H);
        split_qgate(b.qg, b.aq, b.agate, T, NH, HD);
        rmsnorm(b.aq, w.up_f32(ap + ".q_norm.weight"), EPS, b.aq, T * NH, HD);
        rmsnorm(b.ak, w.up_f32(ap + ".k_norm.weight"), EPS, b.ak, T * NKV, HD);
        rope_mrope(b.aq, b.ak, b.pos3d_d, T, NH, NKV, HD, ROT, THETA);
        // append new k,v to the cache, then attend over [0, start_pos+T)
        cudaMemcpy(kc[L] + (int64_t)start_pos * KV_SIZE, b.ak, (int64_t)T * KV_SIZE * 4, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vc[L] + (int64_t)start_pos * KV_SIZE, b.av, (int64_t)T * KV_SIZE * 4, cudaMemcpyDeviceToDevice);
        attention_cached(b.aq, kc[L], vc[L], b.aout, T, NH, NKV, HD, start_pos);
        mul_sigmoid_gate(b.aout, b.agate, b.agated, T * Q_SIZE);
        w.nvfp4(b.agated, ap + ".o_proj", b.mix, T, H, Q_SIZE);
      } else {
        std::string gp = lp + "linear_attn";
        w.bf16(b.xn, gp + ".in_proj_qkv", b.qkv, T, CC, H);
        w.bf16(b.xn, gp + ".in_proj_z", b.z, T, L_V, H);
        w.bf16(b.xn, gp + ".in_proj_a", b.ga, T, L_VH, H);
        w.bf16(b.xn, gp + ".in_proj_b", b.gb, T, L_VH, H);
        causal_conv1d_state_silu(b.qkv, cst[L], (const uint16_t*)w.up_raw(gp + ".conv1d.weight"), b.conv, T, CC, CONV, first);
        cudaMemcpy2D(b.q, L_Q * 4, b.conv, CC * 4, L_Q * 4, T, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(b.k, L_Q * 4, b.conv + L_Q, CC * 4, L_Q * 4, T, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(b.v, L_V * 4, b.conv + 2 * L_Q, CC * 4, L_V * 4, T, cudaMemcpyDeviceToDevice);
        gdn_gating(w.up_f32(gp + ".A_log"), b.ga, b.gb, w.up_f32(gp + ".dt_bias"), b.g, b.beta, T, L_VH);
        gdn_recurrence(b.q, b.k, b.v, b.g, b.beta, scale_gdn, b.recur, gst[L], T, L_KH, L_VH, L_KD, L_VD, first);
        rmsnorm_gated(b.recur, w.up_f32(gp + ".norm.weight"), b.z, EPS, b.gated, T * L_VH, L_VD);
        w.nvfp4(b.gated, gp + ".out_proj", b.mix, T, H, L_V);
      }
      add_inplace(b.h, b.mix, T * H);
      rmsnorm(b.h, w.up_f32(lp + "post_attention_layernorm.weight"), EPS, b.xn2, T, H);
      w.nvfp4(b.xn2, lp + "mlp.gate_proj", b.mg, T, INTER, H);
      w.nvfp4(b.xn2, lp + "mlp.up_proj", b.mu, T, INTER, H);
      swiglu(b.mg, b.mu, b.msw, (int64_t)T * INTER);
      w.nvfp4(b.msw, lp + "mlp.down_proj", b.mix, T, H, INTER);
      add_inplace(b.h, b.mix, T * H);
      if (tdir && (L < 4 || L == NL - 1)) {
        cudaDeviceSynchronize();
        printf("layer %2d (%s): max_rel=%.2e\n", L, is_attn ? "attn" : "gdn ",
               compare(*tdir + "/ref_h" + std::to_string(L) + ".f32", b.h, (int64_t)T * H));
      }
    }
    rmsnorm(b.h, w.up_f32(w.LP + "norm.weight"), EPS, b.xn, T, H);
    if (tdir) printf("final:  max_rel=%.2e\n", compare(*tdir + "/ref_final.f32", b.xn, (int64_t)T * H));
    gemm_bf16(b.xn + (int64_t)(T - 1) * H, (const uint16_t*)w.up_raw("lm_head.weight"), b.logits, 1, VOCAB, H);
    if (tdir) printf("logits: max_rel=%.2e\n", compare(*tdir + "/ref_logits.f32", b.logits, VOCAB));
    return argmax(b.logits, VOCAB);
  }
};

Engine::Engine() : p_(new Impl) {}
Engine::~Engine() { delete p_; }

// Qwen3-VL M-RoPE position ids for a single image. Text tokens get sequential
// 3D-equal positions; the image pads get (t=const, h=row, w=col) offset by the
// preceding text length; text after continues from max+1 (the M-RoPE "compression").
// Returns pos3d [3*n]; sets img_pos/img_cnt (pad run) and next_pos (first decode pos).
static std::vector<int> get_rope_index(const std::vector<int>& ids, int img_tok, int gh, int gw,
                                       int& img_pos, int& img_cnt, int& next_pos) {
  int n = (int)ids.size();
  std::vector<int> p(3 * n);
  int lh = gh / 2, lw = gw / 2, merged = lh * lw;
  int ed = -1;
  for (int i = 0; i < n; ++i) if (ids[i] == img_tok) { ed = i; break; }
  if (ed < 0) {                                       // text only
    for (int a = 0; a < 3; ++a) for (int i = 0; i < n; ++i) p[a * n + i] = i;
    img_pos = -1; img_cnt = 0; next_pos = n; return p;
  }
  for (int a = 0; a < 3; ++a) for (int i = 0; i < ed; ++i) p[a * n + i] = i;   // text before
  for (int m = 0; m < merged; ++m) {                  // image pads
    int idx = ed + m;
    p[0 * n + idx] = ed;                              // t
    p[1 * n + idx] = ed + m / lw;                     // h = row
    p[2 * n + idx] = ed + m % lw;                     // w = col
  }
  int after = ed + merged, st_idx = ed + std::max(std::max(lh - 1, lw - 1), 0) + 1;
  for (int i = after; i < n; ++i) { int off = i - after; for (int a = 0; a < 3; ++a) p[a * n + i] = st_idx + off; }
  img_pos = ed; img_cnt = merged; next_pos = st_idx + (n - after);
  return p;
}

void Engine::load(const std::string& model_dir, int max_ctx) {
  p_->max_ctx = max_ctx;
  p_->mdir = model_dir;
  p_->w.st.open(model_dir + "/model.safetensors");
  p_->b.init(max_ctx);
  p_->alloc_caches();
  p_->vt.load(model_dir);
}

int Engine::encode_image(const std::vector<float>& pixels, int t, int h, int w) {
  int S = t * h * w;
  float* d_pix; cudaMalloc(&d_pix, (int64_t)S * 1536 * 4);
  cudaMemcpy(d_pix, pixels.data(), (int64_t)S * 1536 * 4, cudaMemcpyHostToDevice);
  int merged = t * (h / 2) * (w / 2);
  if (p_->img_embeds) cudaFree(p_->img_embeds);
  cudaMalloc(&p_->img_embeds, (int64_t)merged * H * 4);
  p_->img_merged = p_->vt.encode(d_pix, t, h, w, p_->img_embeds);
  cudaFree(d_pix);
  return p_->img_merged;
}

std::vector<int> Engine::generate_mm(const std::vector<int>& prompt, int img_token_id, int gh, int gw,
                                     int max_new, int eos, const std::function<void(int)>& on_token) {
  std::vector<int> out;
  int n = (int)prompt.size();
  int img_pos, img_cnt, next_pos;
  std::vector<int> pos3d = get_rope_index(prompt, img_token_id, gh, gw, img_pos, img_cnt, next_pos);
  int nxt = p_->forward_tokens(prompt, n, 0, nullptr, pos3d.data(), p_->img_embeds, img_pos, img_cnt);
  cudaDeviceSynchronize();
  int cache_pos = n;
  for (int s = 0; s < max_new; ++s) {
    if (nxt == eos || cache_pos >= p_->max_ctx) break;
    out.push_back(nxt);
    if (on_token) on_token(nxt);
    int cur = nxt, p3[3] = {next_pos, next_pos, next_pos};
    nxt = p_->forward_tokens({cur}, 1, cache_pos, nullptr, p3);
    cudaDeviceSynchronize();
    ++cache_pos; ++next_pos;
  }
  return out;
}

int Engine::next_token(const std::vector<int>& ids) {
  int T = (int)ids.size();
  if (T > p_->max_ctx) T = p_->max_ctx;
  std::vector<int> in(ids.end() - T, ids.end());
  int r = p_->forward_tokens(in, T, 0, nullptr);   // one-shot prefill
  cudaDeviceSynchronize();
  return r;
}

std::vector<int> Engine::generate(const std::vector<int>& prompt, int max_new, int eos,
                                  const std::function<void(int)>& on_token) {
  std::vector<int> out;
  int T = (int)prompt.size();
  if (T > p_->max_ctx) T = p_->max_ctx;
  std::vector<int> pre(prompt.end() - T, prompt.end());
  int nxt = p_->forward_tokens(pre, T, 0, nullptr);   // prefill the prompt, build caches
  cudaDeviceSynchronize();
  int pos = T, steps = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < max_new; ++s) {
    if (nxt == eos || pos >= p_->max_ctx) break;
    out.push_back(nxt);
    if (on_token) on_token(nxt);
    int cur = nxt;
    nxt = p_->forward_tokens({cur}, 1, pos, nullptr);  // incremental decode of one token
    cudaDeviceSynchronize();
    ++pos; ++steps;
  }
  if (steps > 0) {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[decode] %d tok in %.0f ms = %.1f ms/tok (%.2f tok/s)\n",
            steps, ms, ms / steps, 1000.0 * steps / ms);
  }
  return out;
}

int Engine::validate(const std::vector<int>& ids, const std::string& test_dir) {
  int r = p_->forward_tokens(ids, (int)ids.size(), 0, &test_dir);
  cudaDeviceSynchronize();
  return r;
}

int Engine::max_ctx() const { return p_->max_ctx; }

}  // namespace lb
