// Vision-tower kernels for the Qwen3.6-VL ViT (model.visual.*): standard
// LayerNorm (mean-centering, with bias), GELU (tanh + erf), 2D RoPE, full
// bidirectional multi-head attention, and a bias add for the BF16 linears.
#pragma once
#include <cstdint>

namespace lb {

// LayerNorm with bias: y = (x-mean)/sqrt(var+eps) * weight + bias, per row of H.
void layernorm(const float* x, const float* weight, const float* bias, float eps,
               float* y, int M, int H);

// GELU (tanh approximation): 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³))). In place OK.
void gelu_tanh(const float* x, float* y, int64_t n);
// GELU (exact / erf): 0.5 x (1 + erf(x/√2)). Used by the patch merger.
void gelu_erf(const float* x, float* y, int64_t n);

// y[m,n] += bias[n]  (BF16 linear bias, bias uploaded as f32).
void add_bias(float* y, const float* bias, int M, int N);

// Vision 2D RoPE in place on q,k [S,NH,HD]: q = q*cos + rotate_half(q)*sin, with
// cos/sin [S,HD] (rotate_half splits HD into two halves).
void rope_vision(float* q, float* k, const float* cos, const float* sin, int S, int NH, int HD);

// Full bidirectional MHA (no mask): q,k,v [S,NH,HD] -> out [S,NH,HD], scale HD^-0.5.
void vision_attention(const float* q, const float* k, const float* v, float* out,
                      int S, int NH, int HD);

}  // namespace lb
