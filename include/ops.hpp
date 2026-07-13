// Elementwise / reduction forward ops (RMSNorm, SwiGLU; RoPE/softmax to follow).
#pragma once
#include <cstdint>

namespace lb {

// Per-row RMSNorm: y[m,i] = x[m,i] * rsqrt(mean_i(x[m,i]^2) + eps) * weight[i].
void rmsnorm(const float* x, const float* weight, float eps, float* y, int M, int hidden);

// SwiGLU gate: out = silu(gate) * up, silu(x) = x * sigmoid(x).
void swiglu(const float* gate, const float* up, float* out, int64_t n);

}  // namespace lb
