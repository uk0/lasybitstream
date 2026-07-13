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

  // MTP speculative decode: draft `k` tokens per step with the in-checkpoint MTP
  // head and verify them in one main forward. Output is identical to generate()
  // (greedy verify); falls back to generate() if no MTP head is present.
  std::vector<int> generate_spec(const std::vector<int>& prompt, int max_new, int eos, int k = 4);

  // Validation mode: forward once and compare each dumped hidden state against
  // scripts/ref_forward.py's golden files in `test_dir`. Returns argmax.
  int validate(const std::vector<int>& ids, const std::string& test_dir);

  // Aggregate-batching throughput probe: effective tok/s for a forward over M rows
  // (weights read once), i.e. the per-step throughput of a batch of M concurrent decodes.
  double bench(int M, int iters = 3);

  // --- vision / multimodal ---
  // Encode one image's preprocessed patches `pixels` [t*h*w, 1536] into image
  // embeddings held by the engine. Returns the merged image-token count.
  int encode_image(const std::vector<float>& pixels, int t, int h, int w);
  // Generate for a prompt containing a run of `img_token_id` pads; splices the
  // previously encoded image embeddings over the pads and uses image M-RoPE.
  std::vector<int> generate_mm(const std::vector<int>& prompt, int img_token_id, int gh, int gw,
                               int max_new, int eos,
                               const std::function<void(int)>& on_token = nullptr);

  int max_ctx() const;

 private:
  struct Impl;
  Impl* p_ = nullptr;
};

}  // namespace lb
