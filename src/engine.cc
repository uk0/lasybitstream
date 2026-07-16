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
static uint8_t* dalloc_u8(int64_t n) { uint8_t* p; cudaMalloc(&p, n); return p; }

struct Weights {
  SafeTensors st;
  std::map<std::string, void*> cache;
  std::string LP = "model.language_model.";
  ~Weights() { for (auto& kv : cache) cudaFree(kv.second); }   // free uploaded device weights

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
  void free() {
    for (float* p : {h, xn, xn2, mix, state, logits, qkv, z, ga, gb, conv, q, k, v, g, beta, recur,
                     gated, qg, aq, agate, ak, av, aout, agated, mg, mu, msw}) cudaFree(p);
    cudaFree(ids_d); cudaFree(pos3d_d);
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
  std::vector<uint8_t*> kc, vc;                        // per-layer fp8 K/V cache (attn layers)
  std::vector<float*> ksc, vsc, gst, cst;              // per-(token,head) KV scales + GDN state
  VisionTower vt;                                      // model.visual.* module
  float* img_embeds = nullptr;                         // [merged, H] for the current image
  int img_merged = 0;
  std::string mdir;

  ~Impl() {
    if (kc.empty()) return;                            // never loaded
    b.free();
    for (int L = 0; L < NL; ++L) { cudaFree(kc[L]); cudaFree(vc[L]); cudaFree(ksc[L]); cudaFree(vsc[L]);
      cudaFree(gst[L]); cudaFree(cst[L]);
      if (L < (int)gst_bak.size()) { cudaFree(gst_bak[L]); cudaFree(cst_bak[L]); }
      if (L < (int)gdn_snap.size()) { cudaFree(gdn_snap[L]); cudaFree(conv_snap[L]); } }
    for (float* p : {m_emb, m_hn, m_en, m_cat, m_x, m_xn, m_tmp, m_qg, m_q, m_gate, m_k, m_v, m_ao,
                     m_ag, m_mg, m_mu, m_msw, m_kc, m_vc, m_logits_all, m_hlast, m_dh, img_embeds}) cudaFree(p);
    cudaFree(m_id); cudaFree(m_pos);
  }

  void alloc_caches() {
    kc.assign(NL, nullptr); vc.assign(NL, nullptr); ksc.assign(NL, nullptr); vsc.assign(NL, nullptr);
    gst.assign(NL, nullptr); cst.assign(NL, nullptr);
    for (int L = 0; L < NL; ++L) {
      if ((L % FA_INT) == (FA_INT - 1)) {
        kc[L] = dalloc_u8((int64_t)max_ctx * NKV * HD); vc[L] = dalloc_u8((int64_t)max_ctx * NKV * HD);
        ksc[L] = dalloc((int64_t)max_ctx * NKV); vsc[L] = dalloc((int64_t)max_ctx * NKV);
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
                     int img_pos = -1, int img_cnt = 0, std::vector<int>* argmax_all = nullptr,
                     bool snapshot = false) {
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
        store_kv_fp8(b.ak, b.av, kc[L], vc[L], ksc[L], vsc[L], T, NKV, HD, start_pos);
        attention_cached_fp8(b.aq, kc[L], vc[L], ksc[L], vsc[L], b.aout, T, NH, NKV, HD, start_pos);
        mul_sigmoid_gate(b.aout, b.agate, b.agated, T * Q_SIZE);
        w.nvfp4(b.agated, ap + ".o_proj", b.mix, T, H, Q_SIZE);
      } else {
        std::string gp = lp + "linear_attn";
        w.bf16(b.xn, gp + ".in_proj_qkv", b.qkv, T, CC, H);
        w.bf16(b.xn, gp + ".in_proj_z", b.z, T, L_V, H);
        w.bf16(b.xn, gp + ".in_proj_a", b.ga, T, L_VH, H);
        w.bf16(b.xn, gp + ".in_proj_b", b.gb, T, L_VH, H);
        causal_conv1d_state_silu(b.qkv, cst[L], (const uint16_t*)w.up_raw(gp + ".conv1d.weight"), b.conv, T, CC, CONV, first,
                                 snapshot ? conv_snap[L] : nullptr);
        cudaMemcpy2D(b.q, L_Q * 4, b.conv, CC * 4, L_Q * 4, T, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(b.k, L_Q * 4, b.conv + L_Q, CC * 4, L_Q * 4, T, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(b.v, L_V * 4, b.conv + 2 * L_Q, CC * 4, L_V * 4, T, cudaMemcpyDeviceToDevice);
        gdn_gating(w.up_f32(gp + ".A_log"), b.ga, b.gb, w.up_f32(gp + ".dt_bias"), b.g, b.beta, T, L_VH);
        gdn_recurrence(b.q, b.k, b.v, b.g, b.beta, scale_gdn, b.recur, gst[L], T, L_KH, L_VH, L_KD, L_VD, first,
                       snapshot ? gdn_snap[L] : nullptr);
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
    if (argmax_all) {                                  // per-position argmax (MTP verify)
      gemm_bf16(b.xn, (const uint16_t*)w.up_raw("lm_head.weight"), m_logits_all, T, VOCAB, H);
      argmax_all->resize(T);
      for (int t = 0; t < T; ++t) (*argmax_all)[t] = argmax(m_logits_all + (int64_t)t * VOCAB, VOCAB);
      return (*argmax_all)[T - 1];
    }
    gemm_bf16(b.xn + (int64_t)(T - 1) * H, (const uint16_t*)w.up_raw("lm_head.weight"), b.logits, 1, VOCAB, H);
    if (tdir) printf("logits: max_rel=%.2e\n", compare(*tdir + "/ref_logits.f32", b.logits, VOCAB));
    return argmax(b.logits, VOCAB);
  }

  // One batched decode step: B independent sequences, ONE token each (b.ids_d),
  // all at global position `pos`. Uses per-sequence caches; writes logits [B,VOCAB].
  // Every GEMM/norm runs over M=B rows (weights read once per step, amortized over B).
  void decode_batch(int B, int pos, std::vector<uint8_t*>& kcb, std::vector<uint8_t*>& vcb,
                    std::vector<float*>& kscb, std::vector<float*>& vscb,
                    std::vector<float*>& gstb, std::vector<float*>& cstb, float* logits_b, int mc) {
    std::vector<int> pos3(3 * B, pos);
    cudaMemcpy(b.pos3d_d, pos3.data(), 3 * B * 4, cudaMemcpyHostToDevice);
    embed_gather((const uint16_t*)w.up_raw(w.LP + "embed_tokens.weight"), b.ids_d, b.h, B, H);
    float scale_gdn = 1.0f / std::sqrt((float)L_KD);
    for (int L = 0; L < NL; ++L) {
      std::string lp = w.LP + "layers." + std::to_string(L) + ".";
      rmsnorm(b.h, w.up_f32(lp + "input_layernorm.weight"), EPS, b.xn, B, H);
      if ((L % FA_INT) == (FA_INT - 1)) {
        std::string ap = lp + "self_attn";
        w.nvfp4(b.xn, ap + ".q_proj", b.qg, B, NH * 2 * HD, H);
        w.nvfp4(b.xn, ap + ".k_proj", b.ak, B, KV_SIZE, H);
        w.nvfp4(b.xn, ap + ".v_proj", b.av, B, KV_SIZE, H);
        split_qgate(b.qg, b.aq, b.agate, B, NH, HD);
        rmsnorm(b.aq, w.up_f32(ap + ".q_norm.weight"), EPS, b.aq, B * NH, HD);
        rmsnorm(b.ak, w.up_f32(ap + ".k_norm.weight"), EPS, b.ak, B * NKV, HD);
        rope_mrope(b.aq, b.ak, b.pos3d_d, B, NH, NKV, HD, ROT, THETA);
        store_kv_fp8_batched(b.ak, b.av, kcb[L], vcb[L], kscb[L], vscb[L], B, NKV, HD, pos, mc);
        attention_batched_fp8(b.aq, kcb[L], vcb[L], kscb[L], vscb[L], b.aout, B, NH, NKV, HD, pos, mc);
        mul_sigmoid_gate(b.aout, b.agate, b.agated, B * Q_SIZE);
        w.nvfp4(b.agated, ap + ".o_proj", b.mix, B, H, Q_SIZE);
      } else {
        std::string gp = lp + "linear_attn";
        w.bf16(b.xn, gp + ".in_proj_qkv", b.qkv, B, CC, H);
        w.bf16(b.xn, gp + ".in_proj_z", b.z, B, L_V, H);
        w.bf16(b.xn, gp + ".in_proj_a", b.ga, B, L_VH, H);
        w.bf16(b.xn, gp + ".in_proj_b", b.gb, B, L_VH, H);
        conv1d_state_batched(b.qkv, cstb[L], (const uint16_t*)w.up_raw(gp + ".conv1d.weight"), b.conv, B, CC, CONV);
        cudaMemcpy2D(b.q, L_Q * 4, b.conv, CC * 4, L_Q * 4, B, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(b.k, L_Q * 4, b.conv + L_Q, CC * 4, L_Q * 4, B, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(b.v, L_V * 4, b.conv + 2 * L_Q, CC * 4, L_V * 4, B, cudaMemcpyDeviceToDevice);
        gdn_gating(w.up_f32(gp + ".A_log"), b.ga, b.gb, w.up_f32(gp + ".dt_bias"), b.g, b.beta, B, L_VH);
        gdn_recurrence_batched(b.q, b.k, b.v, b.g, b.beta, scale_gdn, b.recur, gstb[L], B, L_KH, L_VH, L_KD, L_VD);
        rmsnorm_gated(b.recur, w.up_f32(gp + ".norm.weight"), b.z, EPS, b.gated, B * L_VH, L_VD);
        w.nvfp4(b.gated, gp + ".out_proj", b.mix, B, H, L_V);
      }
      add_inplace(b.h, b.mix, B * H);
      rmsnorm(b.h, w.up_f32(lp + "post_attention_layernorm.weight"), EPS, b.xn2, B, H);
      w.nvfp4(b.xn2, lp + "mlp.gate_proj", b.mg, B, INTER, H);
      w.nvfp4(b.xn2, lp + "mlp.up_proj", b.mu, B, INTER, H);
      swiglu(b.mg, b.mu, b.msw, (int64_t)B * INTER);
      w.nvfp4(b.msw, lp + "mlp.down_proj", b.mix, B, H, INTER);
      add_inplace(b.h, b.mix, B * H);
    }
    rmsnorm(b.h, w.up_f32(w.LP + "norm.weight"), EPS, b.xn, B, H);
    gemm_bf16(b.xn, (const uint16_t*)w.up_raw("lm_head.weight"), logits_b, B, VOCAB, H);
  }

  // Aggregate decode throughput: B sequences decoded together, `steps` steps each,
  // from a warm context of `ctx` tokens. Returns tokens/s across all B sequences.
  double bench_decode(int B, int ctx, int steps) {
    std::vector<uint8_t*> kcb(NL, nullptr), vcb(NL, nullptr);
    std::vector<float*> kscb(NL, nullptr), vscb(NL, nullptr), gstb(NL, nullptr), cstb(NL, nullptr);
    int mc = ctx + steps + 8;
    for (int L = 0; L < NL; ++L) {
      if ((L % FA_INT) == (FA_INT - 1)) {
        kcb[L] = dalloc_u8((int64_t)B * mc * NKV * HD); vcb[L] = dalloc_u8((int64_t)B * mc * NKV * HD);
        kscb[L] = dalloc((int64_t)B * mc * NKV); vscb[L] = dalloc((int64_t)B * mc * NKV);
      } else {
        gstb[L] = dalloc((int64_t)B * L_VH * L_VD * L_KD); cudaMemset(gstb[L], 0, (int64_t)B * L_VH * L_VD * L_KD * 4);
        cstb[L] = dalloc((int64_t)B * CC * (CONV - 1)); cudaMemset(cstb[L], 0, (int64_t)B * CC * (CONV - 1) * 4);
      }
    }
    float* logits_b = dalloc((int64_t)B * VOCAB);
    std::vector<int> ids(B, 100);
    cudaMemcpy(b.ids_d, ids.data(), B * 4, cudaMemcpyHostToDevice);
    for (int s = 0; s < 3; ++s) decode_batch(B, ctx + s, kcb, vcb, kscb, vscb, gstb, cstb, logits_b, mc);  // warmup
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int s = 0; s < steps; ++s) decode_batch(B, ctx + 3 + s, kcb, vcb, kscb, vscb, gstb, cstb, logits_b, mc);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    if (B > 1) {   // all B sequences are identical here -> every logit row must match (no cross-seq leak)
      int a0 = argmax(logits_b, VOCAB), a1 = argmax(logits_b + (int64_t)(B - 1) * VOCAB, VOCAB);
      printf("       [check: row0=%d row%d=%d %s]  ", a0, B - 1, a1, a0 == a1 ? "OK" : "MISMATCH");
    }
    for (int L = 0; L < NL; ++L) { cudaFree(kcb[L]); cudaFree(vcb[L]); cudaFree(kscb[L]); cudaFree(vscb[L]); cudaFree(gstb[L]); cudaFree(cstb[L]); }
    cudaFree(logits_b);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (double)B * steps / ((double)ms / 1000.0);
  }

  // ---- MTP speculative decoding ----
  // MTP scratch (T=1) + its own attention KV cache.
  float *m_emb=nullptr,*m_hn=nullptr,*m_en=nullptr,*m_cat=nullptr,*m_x=nullptr,*m_xn=nullptr,*m_tmp=nullptr;
  float *m_qg=nullptr,*m_q=nullptr,*m_gate=nullptr,*m_k=nullptr,*m_v=nullptr,*m_ao=nullptr,*m_ag=nullptr;
  float *m_mg=nullptr,*m_mu=nullptr,*m_msw=nullptr,*m_kc=nullptr,*m_vc=nullptr;
  float *m_logits_all=nullptr, *m_hlast=nullptr, *m_dh=nullptr;
  std::vector<float*> gst_bak, cst_bak;                // GDN + conv state snapshots (whole-state)
  std::vector<float*> gdn_snap, conv_snap;             // per-token state snapshots (MTP rollback, no re-advance)
  int *m_id=nullptr,*m_pos=nullptr;
  static const int MAXV = 10;                          // max verify tokens (k+1)
  static const int MAXV_K = MAXV - 1;                  // max draft tokens
  bool mtp_ready=false;

  void alloc_mtp() {
    m_emb=dalloc(H); m_hn=dalloc(H); m_en=dalloc(H); m_cat=dalloc(2*H); m_x=dalloc(H); m_xn=dalloc(H); m_tmp=dalloc(H);
    m_qg=dalloc(NH*2*HD); m_q=dalloc(Q_SIZE); m_gate=dalloc(Q_SIZE); m_k=dalloc(KV_SIZE); m_v=dalloc(KV_SIZE);
    m_ao=dalloc(Q_SIZE); m_ag=dalloc(Q_SIZE); m_mg=dalloc(INTER); m_mu=dalloc(INTER); m_msw=dalloc(INTER);
    m_kc=dalloc((int64_t)max_ctx*KV_SIZE); m_vc=dalloc((int64_t)max_ctx*KV_SIZE);
    cudaMemset(m_kc,0,(int64_t)max_ctx*KV_SIZE*4); cudaMemset(m_vc,0,(int64_t)max_ctx*KV_SIZE*4);
    m_logits_all=dalloc((int64_t)MAXV*VOCAB); m_hlast=dalloc(H); m_dh=dalloc(H);
    cudaMalloc(&m_id,sizeof(int)); cudaMalloc(&m_pos,3*sizeof(int));
    gst_bak.assign(NL,nullptr); cst_bak.assign(NL,nullptr);
    gdn_snap.assign(NL,nullptr); conv_snap.assign(NL,nullptr);
    for (int L=0;L<NL;++L) {
      if ((L%FA_INT)!=(FA_INT-1)) { gst_bak[L]=dalloc((int64_t)L_VH*L_VD*L_KD); cst_bak[L]=dalloc((int64_t)CC*(CONV-1));
        gdn_snap[L]=dalloc((int64_t)MAXV*L_VH*L_VD*L_KD); conv_snap[L]=dalloc((int64_t)MAXV*CC*(CONV-1)); }
    }
    mtp_ready = w.st.has("mtp.fc.weight");
  }
  // Restore the GDN + conv state to the per-token snapshot after accepted-token
  // index `idx` (0-based) — the MTP verify's snapshot, so no re-advance forward is needed.
  void restore_snap(int idx) {
    int64_t ss = (int64_t)L_VH * L_VD * L_KD, cs = (int64_t)CC * (CONV - 1);
    for (int L = 0; L < NL; ++L) if (gdn_snap[L]) {
      cudaMemcpy(gst[L], gdn_snap[L] + idx * ss, ss * 4, cudaMemcpyDeviceToDevice);
      cudaMemcpy(cst[L], conv_snap[L] + idx * cs, cs * 4, cudaMemcpyDeviceToDevice);
    }
  }
  void snap_state(bool save) {   // save/restore GDN recurrent + conv state (for rollback)
    for (int L=0;L<NL;++L) if (gst[L]) {
      float* a = save?gst_bak[L]:gst[L]; float* b2 = save?gst[L]:gst_bak[L];
      cudaMemcpy(a,b2,(int64_t)L_VH*L_VD*L_KD*4,cudaMemcpyDeviceToDevice);
      float* ca = save?cst_bak[L]:cst[L]; float* cb = save?cst[L]:cst_bak[L];
      cudaMemcpy(ca,cb,(int64_t)CC*(CONV-1)*4,cudaMemcpyDeviceToDevice);
    }
  }

  // One MTP step: given the main hidden `h` [1,H] of the token before `last_tok`,
  // and `last_tok` at sequence position `pos`, predict the next draft token and
  // write the MTP hidden into `h_out` [1,H] (for chaining). Writes MTP KV at pos.
  int mtp_step(const float* h, int last_tok, int pos, float* h_out) {
    std::string M = "mtp.";
    cudaMemcpy(m_id, &last_tok, 4, cudaMemcpyHostToDevice);
    embed_gather((const uint16_t*)w.up_raw(w.LP + "embed_tokens.weight"), m_id, m_emb, 1, H);
    rmsnorm(h, w.up_f32(M + "pre_fc_norm_hidden.weight"), EPS, m_hn, 1, H);
    rmsnorm(m_emb, w.up_f32(M + "pre_fc_norm_embedding.weight"), EPS, m_en, 1, H);
    cudaMemcpy(m_cat, m_en, H * 4, cudaMemcpyDeviceToDevice);          // fc = cat([embed, hidden])
    cudaMemcpy(m_cat + H, m_hn, H * 4, cudaMemcpyDeviceToDevice);
    w.bf16(m_cat, M + "fc", m_x, 1, H, 2 * H);
    std::string lp = M + "layers.0.";
    rmsnorm(m_x, w.up_f32(lp + "input_layernorm.weight"), EPS, m_xn, 1, H);
    w.bf16(m_xn, lp + "self_attn.q_proj", m_qg, 1, NH * 2 * HD, H);
    w.bf16(m_xn, lp + "self_attn.k_proj", m_k, 1, KV_SIZE, H);
    w.bf16(m_xn, lp + "self_attn.v_proj", m_v, 1, KV_SIZE, H);
    split_qgate(m_qg, m_q, m_gate, 1, NH, HD);
    rmsnorm(m_q, w.up_f32(lp + "self_attn.q_norm.weight"), EPS, m_q, NH, HD);
    rmsnorm(m_k, w.up_f32(lp + "self_attn.k_norm.weight"), EPS, m_k, NKV, HD);
    int p3[3] = {pos, pos, pos}; cudaMemcpy(m_pos, p3, 12, cudaMemcpyHostToDevice);
    rope_mrope(m_q, m_k, m_pos, 1, NH, NKV, HD, ROT, THETA);
    cudaMemcpy(m_kc + (int64_t)pos * KV_SIZE, m_k, KV_SIZE * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_vc + (int64_t)pos * KV_SIZE, m_v, KV_SIZE * 4, cudaMemcpyDeviceToDevice);
    attention_cached(m_q, m_kc, m_vc, m_ao, 1, NH, NKV, HD, pos);
    mul_sigmoid_gate(m_ao, m_gate, m_ag, Q_SIZE);
    w.bf16(m_ag, lp + "self_attn.o_proj", m_tmp, 1, H, Q_SIZE);
    add_inplace(m_x, m_tmp, H);
    rmsnorm(m_x, w.up_f32(lp + "post_attention_layernorm.weight"), EPS, m_xn, 1, H);
    w.bf16(m_xn, lp + "mlp.gate_proj", m_mg, 1, INTER, H);
    w.bf16(m_xn, lp + "mlp.up_proj", m_mu, 1, INTER, H);
    swiglu(m_mg, m_mu, m_msw, INTER);
    w.bf16(m_msw, lp + "mlp.down_proj", m_tmp, 1, H, INTER);
    add_inplace(m_x, m_tmp, H);
    cudaMemcpy(h_out, m_x, H * 4, cudaMemcpyDeviceToDevice);   // MTP hidden for chaining
    rmsnorm(m_x, w.up_f32(M + "norm.weight"), EPS, m_xn, 1, H);
    w.bf16(m_xn, "lm_head", b.logits, 1, VOCAB, H);            // shared lm_head
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
  p_->alloc_mtp();
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

// MTP speculative decode: draft k tokens with the chained MTP head, verify all in
// one main forward, accept the matching prefix (greedy verify => output identical
// to plain greedy). GDN/conv state is snapshotted and rolled back on rejection.
std::vector<int> Engine::generate_spec(const std::vector<int>& prompt, int max_new, int eos, int k,
                                       const std::function<void(int)>& on_token) {
  if (!p_->mtp_ready || k < 1) return generate(prompt, max_new, eos, on_token);
  std::vector<int> out;
  int T = (int)prompt.size(); if (T > p_->max_ctx) T = p_->max_ctx;
  std::vector<int> pre(prompt.end() - T, prompt.end());
  int t = p_->forward_tokens(pre, T, 0, nullptr);
  cudaDeviceSynchronize();
  // Populate the MTP KV cache over the prompt so drafts can attend to it
  // (b.h still holds the prompt hiddens here). Position 0 has no predecessor.
  for (int i = 1; i < T; ++i) p_->mtp_step(p_->b.h + (int64_t)(i - 1) * H, pre[i], i, p_->m_dh);
  cudaDeviceSynchronize();
  cudaMemcpy(p_->m_hlast, p_->b.h + (int64_t)(T - 1) * H, H * 4, cudaMemcpyDeviceToDevice);
  int pos = T, fwds = 0; bool done = false;
  auto t0 = std::chrono::steady_clock::now();
  while ((int)out.size() < max_new && !done && pos < p_->max_ctx) {
    int kk = std::min(k, Impl::MAXV_K);
    std::vector<int> drafts;
    { const float* hh = p_->m_hlast; int tok = t, dp = pos;
      for (int j = 0; j < kk; ++j) { int d = p_->mtp_step(hh, tok, dp, p_->m_dh); drafts.push_back(d); hh = p_->m_dh; tok = d; ++dp; } }
    std::vector<int> vids; vids.push_back(t); for (int d : drafts) vids.push_back(d);
    int VT = (int)vids.size();
    std::vector<int> p3(3 * VT); for (int a = 0; a < 3; ++a) for (int i = 0; i < VT; ++i) p3[a * VT + i] = pos + i;
    std::vector<int> preds;
    p_->forward_tokens(vids, VT, pos, nullptr, p3.data(), nullptr, -1, 0, &preds, true);  // verify + per-token snapshot
    cudaDeviceSynchronize(); ++fwds;
    out.push_back(t); if (on_token) on_token(t); if (t == eos) { done = true; break; }   // commit t
    int n = 0;
    while (n < (int)drafts.size() && (int)out.size() < max_new && drafts[n] == preds[n]) {
      out.push_back(drafts[n]); if (on_token) on_token(drafts[n]); if (drafts[n] == eos) { done = true; break; } ++n;
    }
    if (done) break;
    int correction = preds[n];                                     // becomes next round's t
    if (n < (int)drafts.size()) p_->restore_snap(n);               // roll state to after token n — NO re-advance forward
    cudaMemcpy(p_->m_hlast, p_->b.h + (int64_t)n * H, H * 4, cudaMemcpyDeviceToDevice);  // h_{pos+n}
    t = correction; pos = pos + n + 1;
  }
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (!out.empty()) fprintf(stderr, "[spec] %zu tok in %.0f ms = %.2f tok/s | %d main forwards (%.2f tok/forward)\n",
                            out.size(), ms, 1000.0 * out.size() / ms, fwds, (double)out.size() / (fwds ? fwds : 1));
  return out;
}

int Engine::validate(const std::vector<int>& ids, const std::string& test_dir) {
  int r = p_->forward_tokens(ids, (int)ids.size(), 0, &test_dir);
  cudaDeviceSynchronize();
  return r;
}

// Isolates the batched-GEMM amortization (the real question for aggregate batching):
// time the whole NVFP4+BF16 weight set for M rows. A batched decode step touches
// every weight once regardless of M, so if the GEMMs amortize, tok/s = M/time scales.
double Engine::bench(int M, int iters) {
  float* x = dalloc((int64_t)M * INTER);
  float* y = dalloc((int64_t)M * VOCAB);               // sized for lm_head's output
  auto run = [&]() {
    for (int L = 0; L < NL; ++L) {
      std::string lp = p_->w.LP + "layers." + std::to_string(L) + ".";
      bool attn = (L % FA_INT) == (FA_INT - 1);
      if (attn) { p_->w.nvfp4(x, lp + "self_attn.q_proj", y, M, NH * 2 * HD, H);
                  p_->w.nvfp4(x, lp + "self_attn.k_proj", y, M, KV_SIZE, H);
                  p_->w.nvfp4(x, lp + "self_attn.v_proj", y, M, KV_SIZE, H);
                  p_->w.nvfp4(x, lp + "self_attn.o_proj", y, M, H, Q_SIZE); }
      else { p_->w.bf16(x, lp + "linear_attn.in_proj_qkv", y, M, L_Q * 2 + L_V, H);
             p_->w.bf16(x, lp + "linear_attn.in_proj_z", y, M, L_V, H);
             p_->w.bf16(x, lp + "linear_attn.in_proj_a", y, M, L_VH, H);
             p_->w.bf16(x, lp + "linear_attn.in_proj_b", y, M, L_VH, H);
             p_->w.nvfp4(x, lp + "linear_attn.out_proj", y, M, H, L_V); }
      p_->w.nvfp4(x, lp + "mlp.gate_proj", y, M, INTER, H);
      p_->w.nvfp4(x, lp + "mlp.up_proj", y, M, INTER, H);
      p_->w.nvfp4(x, lp + "mlp.down_proj", y, M, H, INTER);
    }
    p_->w.bf16(x, "lm_head", y, M, VOCAB, H);            // the untied head (~2.5 GB)
  };
  run(); cudaDeviceSynchronize();                      // warmup
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) run();
  cudaDeviceSynchronize();
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  cudaFree(x); cudaFree(y);
  return 1000.0 * M * iters / ms;                      // aggregate tok/s (GEMM-bound)
}

double Engine::bench_decode(int B, int ctx, int steps) { return p_->bench_decode(B, ctx, steps); }

int Engine::max_ctx() const { return p_->max_ctx; }

}  // namespace lb
