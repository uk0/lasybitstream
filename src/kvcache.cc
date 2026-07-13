#include "kvcache.hpp"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>

namespace lb {

static const int STAGE = 4;

void HybridKVCache::init(int block_tok, int kv_dim, int hot_blocks, int max_pos,
                         const std::string& nvme_path) {
  block_tok_ = block_tok; kv_dim_ = kv_dim; hot_blocks_ = hot_blocks;
  block_elems_ = (int64_t)block_tok * kv_dim;
  cudaMalloc(&gpu_hot_, (int64_t)hot_blocks * block_elems_ * 4);
  cudaMalloc(&gpu_stage_, (int64_t)STAGE * block_elems_ * 4);
  int max_blocks = (max_pos + block_tok - 1) / block_tok;
  nvme_bytes_ = (int64_t)max_blocks * block_elems_ * 4;
  fd_ = ::open(nvme_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd_ < 0) throw std::runtime_error("kvcache: cannot open " + nvme_path);
  if (ftruncate(fd_, nvme_bytes_) != 0) throw std::runtime_error("kvcache: ftruncate failed");
  nvme_ = (float*)mmap(nullptr, nvme_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (nvme_ == MAP_FAILED) throw std::runtime_error("kvcache: mmap failed");
}

HybridKVCache::~HybridKVCache() {
  if (gpu_hot_) cudaFree(gpu_hot_);
  if (gpu_stage_) cudaFree(gpu_stage_);
  if (nvme_ && nvme_ != MAP_FAILED) munmap(nvme_, nvme_bytes_);
  if (fd_ >= 0) close(fd_);
}

void HybridKVCache::append(const float* kv_dev) {
  int cur_block = n_pos_ / block_tok_, off = n_pos_ % block_tok_, slot = cur_block % hot_blocks_;
  if (off == 0 && cur_block >= hot_blocks_) {           // ring wraps: spill the evicted block to NVMe
    int evict = cur_block - hot_blocks_;
    cudaMemcpy(nvme_ + (int64_t)evict * block_elems_, gpu_hot_ + (int64_t)slot * block_elems_,
               block_elems_ * 4, cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(gpu_hot_ + (int64_t)slot * block_elems_ + (int64_t)off * kv_dim_, kv_dev,
             kv_dim_ * 4, cudaMemcpyDeviceToDevice);
  ++n_pos_;
}

const float* HybridKVCache::block_dev(int b) {
  if (hot_has(b)) return gpu_hot_ + (int64_t)(b % hot_blocks_) * block_elems_;
  for (int i = 0; i < STAGE; ++i) if (stage_block_[i] == b) return gpu_stage_ + (int64_t)i * block_elems_;
  int s = stage_slot_; stage_slot_ = (stage_slot_ + 1) % STAGE;     // LRU-ish over stage slots
  cudaMemcpy(gpu_stage_ + (int64_t)s * block_elems_, nvme_ + (int64_t)b * block_elems_,
             block_elems_ * 4, cudaMemcpyHostToDevice);
  stage_block_[s] = b;
  return gpu_stage_ + (int64_t)s * block_elems_;
}

}  // namespace lb
