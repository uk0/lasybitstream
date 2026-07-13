// Qwen3.6-27B (Qwen3_5ForConditionalGeneration) config + checkpoint model:
// parse HF config.json (text_config) + a single safetensors, validate the
// hybrid GDN/full-attention layer stack, the NVFP4 compressed-tensors groups,
// and the MTP head. Loading correctly == the first milestone.
#pragma once
#include "safetensors.hpp"
#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace lb {

enum class LayerKind { LinearAttn, FullAttn };

struct Qwen35Config {
  int64_t num_hidden_layers = 0;
  int64_t hidden_size = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  int64_t vocab_size = 0;
  double partial_rotary_factor = 0;
  double rms_norm_eps = 1e-6;
  int64_t full_attention_interval = 0;
  int64_t linear_num_key_heads = 0;
  int64_t linear_num_value_heads = 0;
  int64_t linear_key_head_dim = 0;
  int64_t linear_value_head_dim = 0;
  int64_t linear_conv_kernel_dim = 0;
  int64_t num_nextn_predict_layers = 0;  // MTP predict layers
  std::vector<LayerKind> layer_types;
  std::string quant_format;              // nvfp4-pack-quantized
  std::string quant_method;              // compressed-tensors
  int64_t quant_group_size = 0;
  int64_t quant_num_bits = 0;
  bool tie_word_embeddings = false;

  static Qwen35Config from_file(const std::string& config_path);
};

// An NVFP4 (compressed-tensors) linear: packed nibbles + per-group fp8 scale +
// per-tensor fp32 global scales for weight and input activations.
struct QuantLinear {
  std::string packed;               // U8  [out, in/2]
  std::string weight_scale;         // F8_E4M3 [out, in/group]
  std::string weight_global_scale;  // F32 [1]
  std::string input_global_scale;   // F32 [1]
  int64_t out = 0, in = 0;
  bool ok = false;
};

struct ValidationReport {
  bool pass = false;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
  int64_t n_linear_layers = 0, n_full_layers = 0;
  int64_t n_quant_groups = 0;   // NVFP4 quantized linears validated
  int64_t n_tensors_seen = 0;
  int64_t mtp_tensors = 0;
  bool mtp_present = false;
  bool vision_present = false;
  uint64_t bytes_total = 0;
  std::map<std::string, int> dtype_counts;
  std::map<std::string, uint64_t> dtype_bytes;
};

class Qwen35Model {
 public:
  Qwen35Config cfg;
  SafeTensors st;
  ValidationReport report;

  // Parse config.json, mmap the safetensors, validate the full structure.
  void load(const std::string& model_dir);
  void print_summary() const;

  // Pick one real NVFP4 weight tensor (for the device round-trip test).
  const QuantLinear* first_quant_linear() const;

 private:
  std::vector<QuantLinear> quant_linears_;  // every validated NVFP4 linear
  QuantLinear check_quant_linear(const std::string& prefix);
  void validate();
};

}  // namespace lb
