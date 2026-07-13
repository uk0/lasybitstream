<p align="center">
  <img src="assets/logo.png" width="320" alt="lasybitstream">
</p>

<h1 align="center">lasybitstream</h1>

<p align="center">
  为 <b>Qwen3.6-27B</b> 定制的 <b>clang + CUDA</b> 手写推理实现，<br>
  跑在 <b>NVIDIA DGX Spark</b>（GB10，sm_121a）上。
</p>

<p align="center"><a href="README_en.md">English</a> · 中文</p>

---

一个从零手写的 CUDA 推理实现，针对 **Qwen3.6-27B**（NVFP4）在 **DGX Spark**（GB10，
128GB 统一内存 @ ~273 GB/s）上定制，不依赖 PyTorch / cuBLAS / CUTLASS。目标只是把这个
模型在这台机器上跑起来并尽量把性能做高一点。`clang++`（host）+ `nvcc`（device，
`-arch=sm_121a`）编译。

## 性能（实测）

| 场景 | 吞吐 |
|---|---|
| 单流解码（贪心，KV 缓存 + 硬件 FP4 解码） | ~6.8 tok/s |
| 聚合批处理峰值（bf16 张量核 GEMM，batch 32） | ~35 tok/s |

单流受显存带宽约束（每 token 读 ~22GB 权重 ÷ 273 GB/s ≈ 12.4 tok/s 上限）；聚合靠一次
权重扫描服务多行来提高吞吐。张量核 GEMM 的进一步流水线优化仍在进行。

## 已验证

每个 kernel 落地前都对标参考做数值验证。

| 组件 | 验证 |
|---|---|
| NVFP4 反量化 / GEMM | 逐位精确 / max_rel 4e-4 |
| GDN 门控 delta-net 递归 | 对标 fla，max_rel 2.5e-4 |
| 完整 64 层前向 | 对标 FP32 金标，贪心 next-token 完全一致 |
| 字节级 BPE 分词器 | 对标 HF 35/35 逐字节精确 + ChatML 模板 |
| 视觉塔前向 | 图像 embedding 对标 HF，端到端可用 |
| MTP 投机解码 | 输出与贪心逐 token 一致 |

## 功能

原生前向（KV + GDN 状态缓存，增量解码）· 文本 / 视觉多模态 · OpenAI + Anthropic 双 API
（流式 + 非流式）· MTP 投机解码 · 聚合批处理。

## 构建与运行

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=121a
cmake --build build -j
./build/lbinfer  /path/to/Qwen3.6-27B-NVFP4 test          # 对标金标校验前向
./build/lbinfer  /path/to/Qwen3.6-27B-NVFP4 test bench    # 聚合吞吐曲线
./build/lbserve  /path/to/Qwen3.6-27B-NVFP4 8080          # OpenAI + Anthropic 服务
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -d \
  '{"messages":[{"role":"user","content":"用一句话解释什么是GPU。"}],"max_tokens":40}'
```

`/v1/chat/completions`（OpenAI）、`/v1/messages`（Anthropic）、`/v1/models`、`/health`，
仅绑 `127.0.0.1`。需要 CUDA 13 + clang，仅在 `sm_121`（GB10）上运行；设备码走
`nvcc -ccbin clang++`（clang 不能直接给 sm_121 出设备码）。
