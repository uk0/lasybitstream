// NVMe + RAM hybrid hot-KV cache. A sequence's K/V is stored in fixed-size blocks;
// the most-recent `hot_blocks` live in GPU (unified) memory, older blocks are spilled
// to an NVMe-backed mmap file and staged back on demand. This lets a single sequence's
// context (or many concurrent sequences) exceed what fits in GPU memory, keeping the
// hot window fast while cold blocks live on disk.
#pragma once
#include <string>
#include <cstdint>
#include <vector>

namespace lb {

class HybridKVCache {
 public:
  // block_tok positions per block; kv_dim floats per position (e.g. NKV*HD for k or v);
  // hot_blocks kept resident in GPU; max_pos upper bound; nvme_path = file on NVMe.
  void init(int block_tok, int kv_dim, int hot_blocks, int max_pos, const std::string& nvme_path);
  ~HybridKVCache();

  // Append one position's kv vector (device ptr [kv_dim]). Write-through: GPU hot ring
  // + NVMe backing. Evicts the oldest hot block to NVMe when the ring wraps.
  void append(const float* kv_dev);

  // Device pointer to block `b`'s data [block_tok * kv_dim]: the GPU hot slot if resident,
  // else staged from NVMe into a device buffer (LRU over the stage slots).
  const float* block_dev(int b);

  int positions() const { return n_pos_; }
  int block_tok() const { return block_tok_; }
  int num_blocks() const { return (n_pos_ + block_tok_ - 1) / block_tok_; }
  // spilled/hot accounting for diagnostics
  int hot_resident() const { return hot_blocks_ < num_blocks() ? hot_blocks_ : num_blocks(); }
  int spilled_blocks() const { int nb = num_blocks(); return nb > hot_blocks_ ? nb - hot_blocks_ : 0; }

 private:
  int block_tok_ = 0, kv_dim_ = 0, hot_blocks_ = 0, n_pos_ = 0;
  int64_t block_elems_ = 0;
  float* gpu_hot_ = nullptr;     // [hot_blocks * block_elems] ring
  float* gpu_stage_ = nullptr;   // [STAGE * block_elems] for cold blocks
  int stage_slot_ = 0, stage_block_[4] = {-1, -1, -1, -1};
  int fd_ = -1;
  float* nvme_ = nullptr;        // mmap'd backing [max_blocks * block_elems]
  int64_t nvme_bytes_ = 0;
  bool hot_has(int b) const { return b >= num_blocks() - hot_blocks_ && b < num_blocks(); }
};

}  // namespace lb
