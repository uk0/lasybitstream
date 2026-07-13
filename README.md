<p align="center">
  <img src="assets/logo.png" width="360" alt="lasybitstream">
</p>

<h1 align="center">lasybitstream</h1>

<p align="center">
  从零手写、专为 <b>Qwen3.6-27B</b> 定制的 <b>clang + CUDA</b> 聚合吞吐推理引擎，<br>
  运行于 <b>NVIDIA DGX Spark</b>（GB10 Grace-Blackwell，sm_121）。
</p>

<p align="center"><a href="README_en.md">English</a> · 中文</p>

---

**lasybitstream** 是一个从零手写的推理引擎——不依赖 PyTorch、cuBLAS、CUTLASS，零依赖
——专为 **DGX Spark**（GB10，`sm_121`，aarch64，128GB 统一 LPDDR5x @ ~273 GB/s）上的
**Qwen3.6-27B**（`Qwen3_5ForConditionalGeneration`，NVFP4）定制。每个 kernel 都用 CUDA
从零手写，`clang++`（host）+ `nvcc`（device，`-arch=sm_121a`）编译。

## 为什么是聚合吞吐

解码受显存带宽约束——每生成一个 token 要把全部激活权重读一遍：

```
Qwen3.6-27B @ NVFP4  ≈ 22 GB 激活权重 / token（含 BF16 GDN 投影 + lm_head）
GB10 带宽            ≈ 273 GB/s
单流上限 = 273 / 22 ≈ 12.4 tok/s
```

单流 50-500 tok/s 对 dense 27B **物理不可行**（片上内存 < 100MB，绕不过 HBM read-once）。
唯一破法是 **批处理内权重复用**：一次权重扫描服务 `B` 行，聚合吞吐随 batch 提升。目标是
冲破 vLLM 基线（~506 tok/s @ batch 128）。

## 模型（Qwen3.5 = Qwen3-Next 架构）

- Dense 27B，64 层：**48 层 GDN 门控 delta-net 线性注意力** + **16 层全 softmax 注意力**
  （3 linear + 1 full）。hidden 5120，24 q / 4 kv 头，head_dim 256，词表 248320，部分 RoPE 0.25。
- **NVFP4**（compressed-tensors）：`weight_packed`(fp4 e2m1) + `weight_scale`(fp8 e4m3，每 16 一组)
  + `weight_global_scale`(fp32)。GDN 输入投影保 BF16。
- **视觉塔** `model.visual.*`（27 层 ViT + 2D RoPE + 2×2 merger 投到 5120）。
- 自带 **MTP** 头用于投机解码。Gemma 式 RMSNorm。

## 已跑通 / 已验证

每个 kernel 落地前都对标参考做数值验证；端到端对标 FP32 金标。

| 组件 | 验证 |
|---|---|
| NVFP4 反量化 / GEMM | 逐位精确 / max_rel 4e-4；**硬件 FP4 解码**（`cuda_fp4.h`，sm_121a） |
| GDN 门控 delta-net | 递归 + gating + gated-norm 对标 fla，max_rel 2.5e-4 |
| **完整 64 层前向** | 对标 FP32 金标：layer-0 2e-4，**贪心 next-token 完全一致** |
| 字节级 BPE 分词器 | 对标 HF **35/35 逐字节精确** + ChatML 模板 |
| **视觉端到端** | 图像 → "This image displays a smooth color gradient…"（正确） |
| **MTP 投机解码** | 输出与贪心逐 token 一致（greedy verify） |
| **聚合批处理** | staged GEMM 摊薄权重读，M=4 达 25.9 tok/s（2.4× 单流） |

**能力**：原生前向（KV + GDN 状态缓存，增量解码）· 文本/视觉多模态 · OpenAI + Anthropic 双 API
（流式 + 非流式）· MTP 投机解码 · 聚合批处理。

## 构建与运行

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=121a    # clang++ host + nvcc device
cmake --build build -j
./build/lbinfer  /path/to/Qwen3.6-27B-NVFP4 test          # 对标金标校验前向
./build/lbinfer  /path/to/Qwen3.6-27B-NVFP4 test bench    # 聚合吞吐曲线
./build/lbtest_tok /path/to/Qwen3.6-27B-NVFP4 test/tok_battery.json  # 分词器
./build/lbserve  /path/to/Qwen3.6-27B-NVFP4 8080         # OpenAI + Anthropic 服务
```

```bash
# OpenAI
curl http://127.0.0.1:8080/v1/chat/completions -d \
  '{"messages":[{"role":"user","content":"用一句话解释什么是GPU。"}],"max_tokens":40}'
# Anthropic
curl http://127.0.0.1:8080/v1/messages -d \
  '{"max_tokens":40,"messages":[{"role":"user","content":"Hi"}]}'
```

`POST /v1/chat/completions`（OpenAI，流式 SSE + 非流式）、`POST /v1/messages`（Anthropic 原生）、
`GET /v1/models`、`/health`，仅绑 `127.0.0.1`。

需要 CUDA 13 + clang，仅在 `sm_121`（GB10）上运行。clang 无法直接给 `sm_121` 出设备码
（只认到 CUDA 12.3），所以设备码走 `nvcc -ccbin clang++`。

## 性能路线（冲 500+ tok/s/卡）

- **张量核 GEMM**：staged f32 批处理已摊薄权重读但受 f32 算力墙（~25 tok/s）；bf16/fp4
  `mma.sync` 张量核把算力墙抬 ~20×，使批处理重回带宽约束（M=64 → ~800 tok/s）。WIP 内核见
  `cuda/nvfp4.cu`（`LB_WMMA=1`）。
- **MTP 加速**：降低草稿开销 + 消除 re-advance（逐 token 状态快照）。
- **NVMe + 内存混合热 KV**：长上下文 / 多并发时冷 KV 落盘。
