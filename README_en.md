<p align="center">
  <img src="assets/logo.png" width="320" alt="lasybitstream">
</p>

<h1 align="center">lasybitstream</h1>

<p align="center">
  A hand-written <b>clang + CUDA</b> inference implementation customized for<br>
  <b>Qwen3.6-27B</b> on the <b>NVIDIA DGX Spark</b> (GB10, sm_121a).
</p>

<p align="center">English · <a href="README.md">中文</a></p>

---

A from-scratch CUDA inference implementation customized for **Qwen3.6-27B** (NVFP4)
on the **DGX Spark** (GB10, 128 GB unified memory @ ~273 GB/s), with no PyTorch /
cuBLAS / CUTLASS. The goal is simply to run this model on this machine and push the
performance a bit. Compiled with `clang++` (host) + `nvcc` (device, `-arch=sm_121a`).

## Performance (measured)

| case | throughput |
|---|---|
| single-stream decode (greedy, KV cache + hardware FP4 decode) | ~7 tok/s |
| aggregate batch (bf16 tensor-core GEMM) | batch 32 → 91, batch 64 → 143, batch 256 → ~216 tok/s |

Single-stream is memory-bandwidth bound (~22 GB of weights read per token ÷ 273 GB/s
≈ 12.4 tok/s ceiling); batching raises throughput by serving multiple rows per weight
sweep. Further pipelining of the tensor-core GEMM (cp.async double-buffering) is in progress.

## Verified

Every kernel is numerically checked against a reference before it lands.

| component | check |
|---|---|
| NVFP4 dequant / GEMM | bit-exact / max_rel 4e-4 |
| GDN gated-delta-net recurrence | vs fla, max_rel 2.5e-4 |
| full 64-layer forward | vs FP32 golden, greedy next-token exact |
| byte-level BPE tokenizer | 35/35 byte-exact vs HF + ChatML template |
| vision tower forward | image embeddings vs HF, end-to-end working |
| MTP speculative decode | output byte-identical to greedy |

## Features

Native forward (KV + GDN-state cache, incremental decode) · text + vision multimodal ·
OpenAI + Anthropic dual API (streaming + non-streaming) · MTP speculative decoding ·
aggregate batching.

## Build & run

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=121a
cmake --build build -j
./build/lbinfer  /path/to/Qwen3.6-27B-NVFP4 test          # validate forward vs golden
./build/lbinfer  /path/to/Qwen3.6-27B-NVFP4 test bench    # aggregate throughput curve
./build/lbserve  /path/to/Qwen3.6-27B-NVFP4 8080          # OpenAI + Anthropic server
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -d \
  '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":16}'
```

`/v1/chat/completions` (OpenAI), `/v1/messages` (Anthropic), `/v1/models`, `/health`,
bound to `127.0.0.1`. Requires CUDA 13 + clang, runs on `sm_121` (GB10) only; device
code goes through `nvcc -ccbin clang++` (clang can't emit sm_121 directly).
