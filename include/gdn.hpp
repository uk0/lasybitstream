// Gated delta-net (GDN) recurrence — the core of Qwen3.5's 48 linear-attention
// layers, matching fla's fused_recurrent_gated_delta_rule.
#pragma once
#include <cstdint>

namespace lb {

// Sequential per-token recurrence. Per v-head, state S[head_v, head_k]:
//   q,k L2-normed; q *= scale; S *= exp(g); v -= S@k; v *= beta;
//   S += outer(v,k); out = S@q.
//   q,k : [T, num_k_heads, head_k]     v : [T, num_v_heads, head_v]
//   g,beta : [T, num_v_heads]          out : [T, num_v_heads, head_v]
// GQA: v-head hv uses k-head hv / (num_v_heads / num_k_heads).
// `state` is a device scratch buffer [num_v_heads * head_v * head_k], zeroed here.
void gdn_recurrence(const float* q, const float* k, const float* v,
                    const float* g, const float* beta, float scale,
                    float* out, float* state,
                    int T, int num_k_heads, int num_v_heads, int head_k, int head_v);

// GDN gating: g = -exp(A_log) * softplus(a + dt_bias),  beta = sigmoid(b).
//   A_log, dt_bias : [num_v_heads]     a, b : [T, num_v_heads]
//   g_out, beta_out : [T, num_v_heads]
void gdn_gating(const float* A_log, const float* a, const float* b, const float* dt_bias,
                float* g_out, float* beta_out, int T, int num_v_heads);

// RMSNormGated (norm_before_gate=True): out = rmsnorm(x)*weight * silu(z), per row.
//   x, z : [M, H]    weight : [H]    out : [M, H]
void rmsnorm_gated(const float* x, const float* weight, const float* z, float eps,
                   float* out, int M, int H);

}  // namespace lb
