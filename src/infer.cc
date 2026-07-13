// Native Qwen3.6-27B forward pass on the GB10. Loads the NVFP4 checkpoint via
// the mmap safetensors reader, uploads weights to the device, and runs the full
// 64-layer decode with the verified kernels (NVFP4 GEMM, Gemma RMSNorm, GDN
// gated-delta-net, partial-RoPE GQA attention, SwiGLU). Compares each dumped
// hidden state against scripts/ref_forward.py and reports the greedy next token.
//
//   ./lbinfer <model_dir> [test_dir]
#include "safetensors.hpp"
#include "nvfp4.hpp"
#include "ops.hpp"
#include "gdn.hpp"
#include "forward.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <cmath>

using namespace lb;

// ---- model dims (Qwen3.6-27B, confirmed from config.json) ----
static const int H = 5120, NL = 64, FA_INT = 4;
static const int NH = 24, NKV = 4, HD = 256;
static const int L_KH = 16, L_VH = 48, L_KD = 128, L_VD = 128, CONV = 4;
static const int ROT = 64, GROUP = 16, VOCAB = 248320;
static const float EPS = 1e-6f, THETA = 1e7f;
static const int Q_SIZE = NH * HD, KV_SIZE = NKV * HD;        // 6144, 1024
static const int L_Q = L_KH * L_KD, L_V = L_VH * L_VD;        // 2048, 6144

static void ck(const char* what) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { printf("CUDA error @ %s: %s\n", what, cudaGetErrorString(e)); }
}

static float* dalloc(int64_t n) { float* p; cudaMalloc(&p, n * sizeof(float)); return p; }

struct Model {
  SafeTensors st;
  std::map<std::string, void*> cache;
  std::string LP = "model.language_model.";

  void load(const std::string& path) { st.open(path + "/model.safetensors"); }

  void* up_raw(const std::string& n) {
    auto it = cache.find(n);
    if (it != cache.end()) return it->second;
    const auto& ti = st.info(n);
    void* d; cudaMalloc(&d, ti.nbytes());
    cudaMemcpy(d, st.data(n), ti.nbytes(), cudaMemcpyHostToDevice);
    cache[n] = d; return d;
  }
  // BF16 tensor -> f32 device buffer (for norm weights, A_log, dt_bias).
  float* up_f32(const std::string& n) {
    std::string key = n + "#f32";
    auto it = cache.find(key);
    if (it != cache.end()) return (float*)it->second;
    const auto& ti = st.info(n);
    int64_t ne = ti.numel();
    const uint16_t* src = (const uint16_t*)st.data(n);
    std::vector<float> host(ne);
    for (int64_t i = 0; i < ne; ++i) {
      uint32_t bits = (uint32_t)src[i] << 16;   // bf16 -> f32
      __builtin_memcpy(&host[i], &bits, 4);
    }
    float* d = dalloc(ne);
    cudaMemcpy(d, host.data(), ne * 4, cudaMemcpyHostToDevice);
    cache[key] = d; return d;
  }
  float gsf(const std::string& n) { return *(const float*)st.data(n); }

  void nvfp4_proj(const float* x, const std::string& pfx, float* y, int M, int64_t out, int64_t in) {
    const uint8_t* packed = (const uint8_t*)up_raw(pfx + ".weight_packed");
    const uint8_t* scale = (const uint8_t*)up_raw(pfx + ".weight_scale");
    gemm_nvfp4(x, packed, scale, gsf(pfx + ".weight_global_scale"), y, M, out, in, GROUP);
  }
  void bf16_proj(const float* x, const std::string& pfx, float* y, int M, int out, int in) {
    gemm_bf16(x, (const uint16_t*)up_raw(pfx + ".weight"), y, M, out, in);
  }
};

// scratch buffers, sized for T tokens
struct Buf {
  float *h, *xn, *xn2, *mix, *state, *logits;
  // gdn
  float *qkv, *z, *ga, *gb, *conv, *q, *k, *v, *g, *beta, *recur, *gated;
  // attn
  float *qg, *aq, *agate, *ak, *av, *aout, *agated;
  // mlp
  float *mg, *mu, *msw;
  int *ids_d, *pos_d;
  void init(int T) {
    h = dalloc((int64_t)T * H); xn = dalloc((int64_t)T * H); xn2 = dalloc((int64_t)T * H);
    mix = dalloc((int64_t)T * H); state = dalloc((int64_t)L_VH * L_VD * L_KD);
    logits = dalloc(VOCAB);
    qkv = dalloc((int64_t)T * (L_Q * 2 + L_V)); z = dalloc((int64_t)T * L_V);
    ga = dalloc((int64_t)T * L_VH); gb = dalloc((int64_t)T * L_VH);
    conv = dalloc((int64_t)T * (L_Q * 2 + L_V));
    q = dalloc((int64_t)T * L_Q); k = dalloc((int64_t)T * L_Q); v = dalloc((int64_t)T * L_V);
    g = dalloc((int64_t)T * L_VH); beta = dalloc((int64_t)T * L_VH);
    recur = dalloc((int64_t)T * L_V); gated = dalloc((int64_t)T * L_V);
    qg = dalloc((int64_t)T * NH * 2 * HD); aq = dalloc((int64_t)T * Q_SIZE); agate = dalloc((int64_t)T * Q_SIZE);
    ak = dalloc((int64_t)T * KV_SIZE); av = dalloc((int64_t)T * KV_SIZE);
    aout = dalloc((int64_t)T * Q_SIZE); agated = dalloc((int64_t)T * Q_SIZE);
    mg = dalloc((int64_t)T * 17408); mu = dalloc((int64_t)T * 17408); msw = dalloc((int64_t)T * 17408);
    cudaMalloc(&ids_d, T * sizeof(int)); cudaMalloc(&pos_d, T * sizeof(int));
  }
};

static double compare(const std::string& ref, const float* dev, int64_t n) {
  FILE* f = fopen(ref.c_str(), "rb");
  if (!f) { printf("  (no ref %s)\n", ref.c_str()); return -1; }
  std::vector<float> r(n); size_t got = fread(r.data(), 4, n, f); fclose(f);
  if ((int64_t)got != n) { printf("  ref size mismatch %s\n", ref.c_str()); return -2; }
  std::vector<float> d(n); cudaMemcpy(d.data(), dev, n * 4, cudaMemcpyDeviceToHost);
  double maxabs = 0, maxrel = 0;
  for (int64_t i = 0; i < n; ++i) {
    double diff = std::fabs((double)d[i] - r[i]);
    maxabs = std::max(maxabs, diff);
    maxrel = std::max(maxrel, diff / (std::fabs((double)r[i]) + 1e-3));
  }
  return maxrel;
}

// One full forward over `ids` (length T). Returns the greedy next-token id.
// When `cmp` is set, each dumped hidden state is compared against the reference.
static int forward(Model& m, Buf& b, const std::vector<int>& ids, int T,
                   const std::string& tdir, bool cmp) {
  std::vector<int> pos(T); for (int i = 0; i < T; ++i) pos[i] = i;
  cudaMemcpy(b.ids_d, ids.data(), T * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.pos_d, pos.data(), T * 4, cudaMemcpyHostToDevice);

  embed_gather((const uint16_t*)m.up_raw(m.LP + "embed_tokens.weight"), b.ids_d, b.h, T, H);
  cudaDeviceSynchronize(); ck("embed");
  if (cmp) printf("embed:  max_rel=%.2e\n", compare(tdir + "/ref_embed.f32", b.h, (int64_t)T * H));

  float scale_gdn = 1.0f / std::sqrt((float)L_KD);
  for (int L = 0; L < NL; ++L) {
    std::string lp = m.LP + "layers." + std::to_string(L) + ".";
    // pre-norm
    rmsnorm(b.h, m.up_f32(lp + "input_layernorm.weight"), EPS, b.xn, T, H);
    bool is_attn = (L % FA_INT) == (FA_INT - 1);
    if (is_attn) {
      std::string ap = lp + "self_attn";
      m.nvfp4_proj(b.xn, ap + ".q_proj", b.qg, T, NH * 2 * HD, H);
      m.nvfp4_proj(b.xn, ap + ".k_proj", b.ak, T, KV_SIZE, H);
      m.nvfp4_proj(b.xn, ap + ".v_proj", b.av, T, KV_SIZE, H);
      split_qgate(b.qg, b.aq, b.agate, T, NH, HD);
      rmsnorm(b.aq, m.up_f32(ap + ".q_norm.weight"), EPS, b.aq, T * NH, HD);
      rmsnorm(b.ak, m.up_f32(ap + ".k_norm.weight"), EPS, b.ak, T * NKV, HD);
      rope_partial(b.aq, b.ak, b.pos_d, T, NH, NKV, HD, ROT, THETA);
      attention_gqa(b.aq, b.ak, b.av, b.aout, T, NH, NKV, HD);
      mul_sigmoid_gate(b.aout, b.agate, b.agated, T * Q_SIZE);
      m.nvfp4_proj(b.agated, ap + ".o_proj", b.mix, T, H, Q_SIZE);
    } else {
      std::string gp = lp + "linear_attn";
      m.bf16_proj(b.xn, gp + ".in_proj_qkv", b.qkv, T, L_Q * 2 + L_V, H);
      m.bf16_proj(b.xn, gp + ".in_proj_z", b.z, T, L_V, H);
      m.bf16_proj(b.xn, gp + ".in_proj_a", b.ga, T, L_VH, H);
      m.bf16_proj(b.xn, gp + ".in_proj_b", b.gb, T, L_VH, H);
      causal_conv1d_silu(b.qkv, (const uint16_t*)m.up_raw(gp + ".conv1d.weight"), b.conv, T, L_Q * 2 + L_V, CONV);
      int C = L_Q * 2 + L_V;
      cudaMemcpy2D(b.q, L_Q * 4, b.conv, C * 4, L_Q * 4, T, cudaMemcpyDeviceToDevice);
      cudaMemcpy2D(b.k, L_Q * 4, b.conv + L_Q, C * 4, L_Q * 4, T, cudaMemcpyDeviceToDevice);
      cudaMemcpy2D(b.v, L_V * 4, b.conv + 2 * L_Q, C * 4, L_V * 4, T, cudaMemcpyDeviceToDevice);
      gdn_gating(m.up_f32(gp + ".A_log"), b.ga, b.gb, m.up_f32(gp + ".dt_bias"), b.g, b.beta, T, L_VH);
      gdn_recurrence(b.q, b.k, b.v, b.g, b.beta, scale_gdn, b.recur, b.state, T, L_KH, L_VH, L_KD, L_VD);
      rmsnorm_gated(b.recur, m.up_f32(gp + ".norm.weight"), b.z, EPS, b.gated, T * L_VH, L_VD);
      m.nvfp4_proj(b.gated, gp + ".out_proj", b.mix, T, H, L_V);
    }
    add_inplace(b.h, b.mix, T * H);
    // mlp
    rmsnorm(b.h, m.up_f32(lp + "post_attention_layernorm.weight"), EPS, b.xn2, T, H);
    m.nvfp4_proj(b.xn2, lp + "mlp.gate_proj", b.mg, T, 17408, H);
    m.nvfp4_proj(b.xn2, lp + "mlp.up_proj", b.mu, T, 17408, H);
    swiglu(b.mg, b.mu, b.msw, (int64_t)T * 17408);
    m.nvfp4_proj(b.msw, lp + "mlp.down_proj", b.mix, T, H, 17408);
    add_inplace(b.h, b.mix, T * H);
    cudaDeviceSynchronize();
    if (cmp && (L < 4 || L == NL - 1)) {
      cudaError_t e = cudaGetLastError();
      double mr = compare(tdir + "/ref_h" + std::to_string(L) + ".f32", b.h, (int64_t)T * H);
      printf("layer %2d (%s): max_rel=%.2e%s\n", L, is_attn ? "attn" : "gdn ", mr,
             e != cudaSuccess ? cudaGetErrorString(e) : "");
    }
  }

  rmsnorm(b.h, m.up_f32(m.LP + "norm.weight"), EPS, b.xn, T, H);
  cudaDeviceSynchronize();
  if (cmp) printf("final:  max_rel=%.2e\n", compare(tdir + "/ref_final.f32", b.xn, (int64_t)T * H));

  gemm_bf16(b.xn + (int64_t)(T - 1) * H, (const uint16_t*)m.up_raw("lm_head.weight"), b.logits, 1, VOCAB, H);
  cudaDeviceSynchronize(); ck("lm_head");
  if (cmp) printf("logits: max_rel=%.2e\n", compare(tdir + "/ref_logits.f32", b.logits, VOCAB));
  return argmax(b.logits, VOCAB);
}

int main(int argc, char** argv) {
  std::string mdir = argc > 1 ? argv[1] : "/model";
  std::string tdir = argc > 2 ? argv[2] : "test";
  int gen = 0;                                  // ./lbinfer <model> <test> gen <N>
  for (int i = 3; i < argc; ++i)
    if (std::string(argv[i]) == "gen" && i + 1 < argc) gen = atoi(argv[i + 1]);
  Model m; m.load(mdir);
  printf("loaded %zu tensors\n", m.st.count());

  std::vector<int> ids;
  { FILE* f = fopen((tdir + "/ref_tokens.i32").c_str(), "rb");
    if (!f) { printf("need %s/ref_tokens.i32\n", tdir.c_str()); return 2; }
    int x; while (fread(&x, 4, 1, f) == 1) ids.push_back(x); fclose(f); }
  printf("prompt T=%zu:", ids.size()); for (int i : ids) printf(" %d", i); printf("\n");

  Buf b; b.init(ids.size() + std::max(gen, 1) + 1);

  if (gen > 0) {
    for (int s = 0; s < gen; ++s) {
      int nxt = forward(m, b, ids, ids.size(), tdir, false);
      ids.push_back(nxt);
      printf("step %2d -> %d\n", s, nxt);
    }
    FILE* f = fopen((tdir + "/gen_ids.i32").c_str(), "wb");
    fwrite(ids.data(), 4, ids.size(), f); fclose(f);
    printf("\ngenerated ids:"); for (int i : ids) printf(" %d", i);
    printf("\nwrote %s/gen_ids.i32\n", tdir.c_str());
  } else {
    int am = forward(m, b, ids, ids.size(), tdir, true);
    printf("\nargmax = %d  (ref expects the same)\n", am);
  }
  return 0;
}
