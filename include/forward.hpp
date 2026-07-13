// Remaining native kernels for the Qwen3.6-27B forward pass. Together with
// nvfp4.hpp (dequant+GEMM), ops.hpp (Gemma RMSNorm, SwiGLU) and gdn.hpp
// (recurrence, gating, gated-norm) these cover every op in the model.
#pragma once
#include <cstdint>

namespace lb {

// BF16 (device __nv_bfloat16 stored as uint16) dense GEMM: y[M,out] = x[M,in] @ W[out,in]^T.
void gemm_bf16(const float* x, const uint16_t* w_bf16, float* y, int M, int out, int in);

// Embedding gather: y[T,H] = embed[ids[t], :]  (embed is BF16 [vocab,H]).
void embed_gather(const uint16_t* embed_bf16, const int* ids, float* y, int T, int H);

// Depthwise causal conv1d (kernel K) over channels, then SiLU.
//   x : [T, C]   w_bf16 : [C, K]   out : [T, C]
//   out[t,c] = silu( sum_{j=0..K-1} w[c,j] * x[t-(K-1)+j, c] ), x=0 for t<0.
void causal_conv1d_silu(const float* x, const uint16_t* w_bf16, float* out, int T, int C, int K);

// Partial NeoX RoPE in-place on q [T,NH,HD] and k [T,NKV,HD].
// Rotates the first `rot` dims (theta base), leaves [rot,HD) untouched.
void rope_partial(float* q, float* k, const int* pos, int T, int NH, int NKV, int HD,
                  int rot, float theta);

// GQA causal self-attention. q [T,NH,HD], k/v [T,NKV,HD] -> out [T,NH,HD].
// scale = HD^-0.5, causal mask, NH/NKV query heads share each kv head.
void attention_gqa(const float* q, const float* k, const float* v, float* out,
                   int T, int NH, int NKV, int HD);

// Cached (incremental) GQA attention: T new query tokens attend over a K/V cache
// of length start_pos+T. Query i (global position start_pos+i) attends keys
// 0..start_pos+i (causal). kc/vc are [start_pos+T, NKV, HD] (k already RoPE'd).
void attention_cached(const float* q, const float* kc, const float* vc, float* out,
                      int T, int NH, int NKV, int HD, int start_pos);

// Depthwise causal conv1d (kernel K) + SiLU with a carried state. `state` holds
// the previous K-1 inputs per channel [C,K-1] (updated in place); when `first`
// the state is treated as zero (fresh sequence). x [T,C] -> out [T,C].
void causal_conv1d_state_silu(const float* x, float* state, const uint16_t* w_bf16,
                              float* out, int T, int C, int K, bool first);

// Per-head split of the fused q/gate projection: in [T,NH,2*HD] (per head [q(HD)|gate(HD)])
// -> q [T,NH,HD], gate [T,NH,HD].
void split_qgate(const float* qg, float* q, float* gate, int T, int NH, int HD);

// out[i] = sigmoid(gate[i]) * x[i], elementwise (attention output gate).
void mul_sigmoid_gate(const float* x, const float* gate, float* out, int n);

// Residual add in place: a[i] += b[i].
void add_inplace(float* a, const float* b, int n);

// argmax over a length-n vector; returns the index (host int).
int argmax(const float* x_dev, int n);

}  // namespace lb
