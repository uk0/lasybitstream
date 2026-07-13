<p align="center">
  <img src="assets/logo.png" width="360" alt="lasybitstream">
</p>

<h1 align="center">lasybitstream</h1>

<p align="center">
  A from-scratch <b>clang + CUDA</b> aggregate-throughput inference engine,
  custom-built for <b>Qwen3.6-27B</b> on the <b>NVIDIA DGX Spark</b> (GB10, sm_121).<br>
  <i>从零手写、专为 Qwen3.6-27B 定制的 clang+cuda 高吞吐推理引擎。</i>
</p>

---

## English

**lasybitstream** is a hand-written inference engine — no PyTorch, no cuBLAS, no
dependencies — for **Qwen3.6-27B** (`Qwen3_5ForConditionalGeneration`, NVFP4) on
the **DGX Spark** (GB10 Grace-Blackwell, `sm_121`, aarch64, 128 GB unified
LPDDR5x @ ~273 GB/s). Every kernel is written from scratch in CUDA, compiled with
`clang++` (host) + `nvcc` (device, `-arch=sm_121`).

### Why aggregate throughput

Decode is memory-bandwidth bound — one generated token reads every active weight
once:

```
Qwen3.6-27B @ NVFP4  ≈ 13.5 GB active weights / token
GB10 bandwidth       ≈ 273 GB/s
single-stream ceiling = 273 / 13.5 ≈ 20 tok/s
```

500 tok/s single-stream would need ~13 GB resident on-chip, but total on-chip
memory (L2 + shared + registers) is < 100 MB — so "baking weights into a
bitstream" (SASS immediates / constant memory / persistent kernels) **cannot**
evade the HBM read-once cost; the model is ~135× too large. Single-stream 500
tok/s is physically impossible for this dense model.

The wall is beaten the only way it can be — **weight reuse across a batch**. One
weight sweep serves `B` rows, so aggregate throughput scales with batch. The goal
is to beat the vLLM baseline (~506 tok/s @ batch 128) with tighter NVFP4 GEMM, a
fused GDN state kernel, and CUDA-graph decode.

### Model (Qwen3.5 = Qwen3-Next arch)

- Dense 27B, 64 layers: **48 GDN (gated delta-net) linear-attention** + **16 full
  softmax-attention**, pattern 3-linear-then-1-full. hidden 5120, 24 q / 4 kv
  heads, head_dim 256, vocab 248320, partial RoPE 0.25.
- **NVFP4** (compressed-tensors): `weight_packed` (fp4 e2m1) + `weight_scale`
  (fp8 e4m3, per-16) + `weight_global_scale` (fp32). GDN in-projections stay BF16.
- Ships an **MTP** head for speculative decoding. Gemma-style RMSNorm.

### Verified so far

Every kernel is numerically matched against a reference before it lands.

| component | check |
|---|---|
| checkpoint loader | loads Qwen3.6-27B-MTP (64 layers, 304 NVFP4 groups, MTP), PASS |
| NVFP4 dequant | **bit-exact** vs `compressed_tensors` (max_abs = 0) |
| NVFP4 GEMM (Linear) | vs CPU matmul, max_rel 4e-4 |
| RMSNorm (Gemma) + SwiGLU | bit-exact / 5e-7 |
| MLP block | `down(silu(gate(x))·up(x))` vs numpy on real weights, max_rel 1e-3 |

**In progress:** full-attention kernel (RoPE + GQA + q/k-norm + output gate) → GDN
gated-delta-net recurrence → 64-layer forward → sampling → tokenizer → matching
vLLM's next-token = working inference.

### Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=121   # clang++ host + nvcc device
cmake --build build -j
./build/lbload  /path/to/Qwen3.6-27B-NVFP4       # load + validate the checkpoint
ctest --test-dir build                           # or run ./build/lbtest_*
```

Requires CUDA 13 + clang. Runs on `sm_121` (GB10) only. `clang` cannot emit
`sm_121` device code directly (knows ≤ CUDA 12.3), so device code goes through
`nvcc -ccbin clang++`.

---

## 中文

**lasybitstream** 是一个从零手写的推理引擎——不依赖 PyTorch、不依赖 cuBLAS、
零依赖——专为 **DGX Spark**（GB10 Grace-Blackwell，`sm_121`，aarch64，128GB 统一
LPDDR5x @ ~273 GB/s）上的 **Qwen3.6-27B** 定制。每个 kernel 都用 CUDA 从零手写，
`clang++`（host）+ `nvcc`（device，`-arch=sm_121`）编译。

### 为什么是聚合吞吐

解码受显存带宽约束——每生成一个 token 都要把全部激活权重读一遍：

```
Qwen3.6-27B @ NVFP4  ≈ 13.5 GB 激活权重 / token
GB10 带宽            ≈ 273 GB/s
单流上限 = 273 / 13.5 ≈ 20 tok/s
```

单流 500 tok/s 需要把 ~13 GB 常驻片上，但片上总内存（L2+共享+寄存器）< 100 MB
——所以"把权重烤进 bitstream"（SASS 立即数 / constant memory / persistent
kernel）**绕不过** HBM 的 read-once 下限，模型大了约 135 倍。单流 500 tok/s 对这个
dense 模型物理上不可行。

显存墙只有一个破法——**批处理内权重复用**：一次权重扫描服务 `B` 行，聚合吞吐随
batch 提升。目标是用更紧的 NVFP4 GEMM + 融合 GDN 状态 kernel + CUDA-graph 解码，
冲破 vLLM 基线（~506 tok/s @ batch 128）。

### 模型（Qwen3.5 = Qwen3-Next 架构）

- Dense 27B，64 层：**48 层 GDN 门控 delta-net 线性注意力** + **16 层全 softmax
  注意力**，3 linear + 1 full 循环。hidden 5120，24 q / 4 kv 头，head_dim 256，
  词表 248320，部分 RoPE 0.25。
- **NVFP4**（compressed-tensors）：`weight_packed`(fp4 e2m1) + `weight_scale`
  (fp8 e4m3，每 16 一组) + `weight_global_scale`(fp32)。GDN 输入投影保 BF16。
- 自带 **MTP** 头用于投机解码。Gemma 式 RMSNorm。

### 已验证组件

每个 kernel 落地前都对标参考做数值验证。

| 组件 | 验证 |
|---|---|
| checkpoint 加载器 | 加载 Qwen3.6-27B-MTP（64 层，304 NVFP4 组，MTP），PASS |
| NVFP4 反量化 | 对标 `compressed_tensors` **逐位精确**（max_abs = 0） |
| NVFP4 GEMM（Linear） | 对标 CPU 矩阵乘，max_rel 4e-4 |
| RMSNorm（Gemma）+ SwiGLU | 逐位精确 / 5e-7 |
| MLP 块 | `down(silu(gate(x))·up(x))` 对标 numpy 真权重，max_rel 1e-3 |

**进行中**：全注意力 kernel（RoPE + GQA + q/k-norm + output gate）→ GDN 门控
delta-net 递归 → 64 层前向 → 采样 → tokenizer → 对上 vLLM 的 next-token = 可推理。

### 构建

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=121   # clang++ host + nvcc device
cmake --build build -j
./build/lbload  /path/to/Qwen3.6-27B-NVFP4       # 加载+校验 checkpoint
```

需要 CUDA 13 + clang，仅在 `sm_121`（GB10）上运行。clang 无法直接给 `sm_121`
出设备码（只认到 CUDA 12.3），所以设备码走 `nvcc -ccbin clang++`。
