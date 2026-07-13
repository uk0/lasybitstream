# lasybitstream

A from-scratch **aggregate-throughput** inference engine for **Qwen3.6-27B** on the
**NVIDIA DGX Spark** (GB10 Grace-Blackwell, `sm_121`, aarch64, 128 GB unified
LPDDR5x @ ~273 GB/s), built with **clang + CUDA**.

## Why aggregate, not single-stream

Decode is memory-bandwidth bound. One generated token must read every active
weight once:

```
Qwen3.6-27B @ NVFP4  ≈ 13.5 GB active weights / token
GB10 bandwidth       ≈ 273 GB/s
single-stream ceiling = 273 / 13.5 ≈ 20 tok/s     (measured: ~19)
```

500 tok/s single-stream would need 6.75 TB/s, i.e. ~13 GB resident on-chip — but
total on-chip memory (L2 + all SM shared memory + registers + constant) is
< 100 MB. **No amount of "baking weights into the kernel" (SASS immediates,
constant memory, persistent kernels) evades the HBM read-once cost**; the model
is ~135× too large. Single-stream 500 tok/s is physically impossible for this
dense model; an FPGA would stream from HBM for the same reason.

The bandwidth wall is beaten the only way it can be — **weight reuse across a
batch**. One weight sweep serves `B` independent rows:

```
aggregate ceiling ≈ B × 20 tok/s     (bandwidth-only; compute caps it lower)
batch 64  → measured ~426 tok/s (vLLM baseline)
batch 128 → measured ~506 tok/s
```

**Goal: beat the vLLM baseline (~506 tok/s @128) with tighter NVFP4 GEMM, a fused
GDN state kernel, CUDA-graph decode, and MTP verification** — while staying
honest about accuracy (see below).

## Accuracy stance

Quantized / batched / GPU kernels are **not byte-identical** to single-token
greedy: shape-dependent rounding can flip near-tied argmaxes (colibri documents
the same, and ~0.3% RMS error per int-dot matmul). Every emitted token is still
the argmax of a *valid* forward, so continuations stay correct — but the stream
may differ from a reference greedy run. A separate `--exact` conformance mode
(reference kernels, no speculation, fixed reduction order) is provided for
byte-exact verification; it is not the throughput path.

## Target architecture (Qwen3_5ForConditionalGeneration)

- Dense 27B, 64 layers: **48 GDN (gated delta-net) linear-attention** + **16 full
  softmax-attention**. 256K context. `partial_rotary_factor = 0.25`.
- NVFP4 (compressed-tensors) weights; ships an MTP head for speculative decode.
- GDN layers keep a fixed-size recurrent state (no growing KV) → they cut KV
  traffic, not the dominant 13.5 GB weight sweep. GDN projections are precision
  sensitive — keep them off aggressive FP4 where the checkpoint intends higher
  precision.

## Component map

| path | role |
|---|---|
| `CMakeLists.txt` | aarch64 + CUDA 13 + `sm_121`; clang host / nvcc device |
| `cuda/nvfp4_gemm.cu` | NVFP4 batched projections for `sm_121` (decode + verify shapes) |
| `cuda/gdn_decode.cu` | fused GDN recurrent state update |
| `cuda/mtp_verify.cu` | draft + batch-verify + rejection/rollback |
| `src/model.cc` | checkpoint loader: config, compressed-tensors NVFP4, layer types, MTP |
| `src/engine.cc` | runtime: GDN state, KV for softmax layers, continuous batching, CUDA graphs |
| `bench/` | bandwidth + batched-GEMM microbenchmarks (baseline the ceiling) |

colibri's headline win (streaming MoE experts from disk) does **not** apply to a
dense model and is intentionally not ported. What transfers: MTP speculative
decode, integer/low-bit quant kernels, KV compression, batch-union verification.

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=121 && cmake --build build -j
```

Requires CUDA 13 + clang. Runs on `sm_121` (GB10) only.

## Status

Bring-up: toolchain + microbenchmarks establishing the measured bandwidth/compute
ceiling before engine components land.
