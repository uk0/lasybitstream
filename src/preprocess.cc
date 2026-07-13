#include "preprocess.hpp"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace lb {

std::vector<uint8_t> base64_decode(const std::string& s) {
  static int8_t T[256]; static bool init = false;
  if (!init) { std::memset(T, -1, 256);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) T[(uint8_t)a[i]] = (int8_t)i; init = true; }
  std::vector<uint8_t> out; int val = 0, bits = 0;
  for (char c : s) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
    int8_t d = T[(uint8_t)c]; if (d < 0) continue;
    val = (val << 6) | d; bits += 6;
    if (bits >= 8) { bits -= 8; out.push_back((uint8_t)((val >> bits) & 0xff)); }
  }
  return out;
}

// Qwen smart-resize: round to a multiple of factor within [min,max] pixels.
static void smart_resize(int h, int w, int factor, int64_t minp, int64_t maxp, int& hb, int& wb) {
  auto rnd = [&](double v) { return (int)std::max((double)factor, std::round(v / factor) * factor); };
  hb = rnd(h); wb = rnd(w);
  if ((int64_t)hb * wb > maxp) { double beta = std::sqrt((double)h * w / maxp);
    hb = std::max(factor, (int)std::floor(h / beta / factor) * factor);
    wb = std::max(factor, (int)std::floor(w / beta / factor) * factor); }
  else if ((int64_t)hb * wb < minp) { double beta = std::sqrt((double)minp / ((double)h * w));
    hb = (int)std::ceil(h * beta / factor) * factor; wb = (int)std::ceil(w * beta / factor) * factor; }
}

PreImage preprocess_image(const uint8_t* bytes, size_t n) {
  PreImage r;
  int W0, H0, C0;
  uint8_t* img = stbi_load_from_memory(bytes, (int)n, &W0, &H0, &C0, 3);   // force RGB
  if (!img) return r;
  const int P = 16, MERGE = 2, FACTOR = P * MERGE;                        // 32
  int Hr, Wr;
  smart_resize(H0, W0, FACTOR, 3136, 1003520, Hr, Wr);
  int gh = Hr / P, gw = Wr / P, merged_h = gh / MERGE, merged_w = gw / MERGE;
  r.t = 1; r.h = gh; r.w = gw;
  // bilinear resize to [Hr,Wr,3], normalized (x/255 - 0.5)/0.5 = x/127.5 - 1
  std::vector<float> norm((size_t)Hr * Wr * 3);
  for (int y = 0; y < Hr; ++y) for (int x = 0; x < Wr; ++x) {
    float sy = (y + 0.5f) * H0 / Hr - 0.5f, sx = (x + 0.5f) * W0 / Wr - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx);
    float fy = sy - y0, fx = sx - x0;
    for (int c = 0; c < 3; ++c) {
      auto px = [&](int yy, int xx) {
        yy = yy < 0 ? 0 : (yy >= H0 ? H0 - 1 : yy); xx = xx < 0 ? 0 : (xx >= W0 ? W0 - 1 : xx);
        return (float)img[((size_t)yy * W0 + xx) * 3 + c]; };
      float v = (1 - fy) * ((1 - fx) * px(y0, x0) + fx * px(y0, x0 + 1)) +
                fy * ((1 - fx) * px(y0 + 1, x0) + fx * px(y0 + 1, x0 + 1));
      norm[((size_t)y * Wr + x) * 3 + c] = v / 127.5f - 1.f;
    }
  }
  stbi_image_free(img);
  // patchify to 2x2-block order: row = [t, block_h, block_w, ih, iw]; content = [c, temporal, py, px]
  int S = gh * gw;
  r.pixels.resize((size_t)S * 1536);
  for (int bh = 0; bh < merged_h; ++bh) for (int bw = 0; bw < merged_w; ++bw)
    for (int ih = 0; ih < MERGE; ++ih) for (int iw = 0; iw < MERGE; ++iw) {
      int row = ((bh * merged_w + bw) * MERGE + ih) * MERGE + iw;
      int ph = bh * MERGE + ih, pw = bw * MERGE + iw;
      float* dst = &r.pixels[(size_t)row * 1536];
      for (int c = 0; c < 3; ++c) for (int tt = 0; tt < 2; ++tt) for (int py = 0; py < P; ++py) for (int px = 0; px < P; ++px)
        dst[((c * 2 + tt) * P + py) * P + px] = norm[(((size_t)(ph * P + py) * Wr) + (pw * P + px)) * 3 + c];
    }
  r.ok = true;
  return r;
}

}  // namespace lb
