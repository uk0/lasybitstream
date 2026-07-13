// Zero-copy safetensors reader: mmap the file, parse the header, expose a
// tensor table (name -> dtype/shape/byte-range) and pointers into the blob.
#pragma once
#include "json.hpp"
#include <string>
#include <map>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace lb {

struct TensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;  // byte range within the data blob
  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return shape.empty() ? 0 : n;
  }
  uint64_t nbytes() const { return end - begin; }
};

inline size_t dtype_bytes(const std::string& dt) {
  if (dt == "F64" || dt == "I64" || dt == "U64") return 8;
  if (dt == "F32" || dt == "I32" || dt == "U32") return 4;
  if (dt == "F16" || dt == "BF16" || dt == "I16" || dt == "U16") return 2;
  if (dt == "F8_E4M3" || dt == "F8_E5M2" || dt == "I8" || dt == "U8" || dt == "BOOL") return 1;
  return 0;  // unknown / sub-byte packed
}

class SafeTensors {
  int fd_ = -1;
  void* map_ = nullptr;
  size_t filesize_ = 0;
  const uint8_t* blob_ = nullptr;  // start of tensor data
  std::map<std::string, TensorInfo> tensors_;

 public:
  ~SafeTensors() {
    if (map_ && map_ != MAP_FAILED) munmap(map_, filesize_);
    if (fd_ >= 0) close(fd_);
  }

  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
  size_t filesize() const { return filesize_; }
  size_t count() const { return tensors_.size(); }
  bool has(const std::string& n) const { return tensors_.count(n); }
  const TensorInfo& info(const std::string& n) const { return tensors_.at(n); }

  // Pointer to a tensor's bytes inside the mmapped blob.
  const uint8_t* data(const std::string& n) const {
    return blob_ + tensors_.at(n).begin;
  }

  void open(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("safetensors: cannot open " + path);
    struct stat st{};
    if (fstat(fd_, &st) != 0) throw std::runtime_error("safetensors: fstat failed");
    filesize_ = (size_t)st.st_size;
    if (filesize_ < 8) throw std::runtime_error("safetensors: file too small");
    map_ = mmap(nullptr, filesize_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) throw std::runtime_error("safetensors: mmap failed");
    const uint8_t* base = (const uint8_t*)map_;

    uint64_t header_len;
    __builtin_memcpy(&header_len, base, 8);  // little-endian u64
    if (8 + header_len > filesize_) throw std::runtime_error("safetensors: header overruns file");

    Json h = json_parse((const char*)base + 8, header_len);
    if (!h.is_obj()) throw std::runtime_error("safetensors: header is not an object");
    blob_ = base + 8 + header_len;
    uint64_t blob_bytes = filesize_ - 8 - header_len;

    for (auto& kv : h.obj) {
      if (kv.first == "__metadata__") continue;
      const Json& v = kv.second;
      TensorInfo t;
      t.dtype = v.at("dtype").s();
      for (auto& d : v.at("shape").arr) t.shape.push_back(d.i64());
      const Json& off = v.at("data_offsets");
      if (off.arr.size() == 2) { t.begin = off.arr[0].i64(); t.end = off.arr[1].i64(); }
      if (t.end > blob_bytes || t.begin > t.end)
        throw std::runtime_error("safetensors: tensor '" + kv.first + "' offsets out of range");
      tensors_[kv.first] = std::move(t);
    }
  }
};

}  // namespace lb
