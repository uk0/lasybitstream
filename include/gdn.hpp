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

}  // namespace lb
