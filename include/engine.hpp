// The Qwen3.6-27B inference engine: loads the NVFP4 checkpoint and runs the
// native forward pass. Hides all CUDA/loader details behind a pImpl so the CLI
// and the HTTP server can share it without pulling in CUDA headers.
#pragma once
#include <string>
#include <vector>
#include <functional>

namespace lb {

class Engine {
 public:
  Engine();
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  void load(const std::string& model_dir, int max_ctx = 4096);

  // Greedy next-token id for the given context (full prefill).
  int next_token(const std::vector<int>& ids);

  // Greedy decode up to max_new tokens, stopping at `eos`. `on_token`, if set,
  // is called with each new id as it is produced (for streaming). Returns the
  // generated ids (excluding the prompt).
  std::vector<int> generate(const std::vector<int>& prompt, int max_new, int eos,
                            const std::function<void(int)>& on_token = nullptr);

  // Validation mode: forward once and compare each dumped hidden state against
  // scripts/ref_forward.py's golden files in `test_dir`. Returns argmax.
  int validate(const std::vector<int>& ids, const std::string& test_dir);

  int max_ctx() const;

 private:
  struct Impl;
  Impl* p_ = nullptr;
};

}  // namespace lb
