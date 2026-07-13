#include "qwen35.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>

namespace lb {

static std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

Qwen35Config Qwen35Config::from_file(const std::string& path) {
  std::string txt = read_file(path);
  Json root = json_parse(txt);
  // Qwen3_5ForConditionalGeneration nests the LM under text_config.
  const Json& tc = root.has("text_config") ? root.at("text_config") : root;

  Qwen35Config c;
  c.num_hidden_layers      = tc.at("num_hidden_layers").i64();
  c.hidden_size            = tc.at("hidden_size").i64();
  c.num_attention_heads    = tc.at("num_attention_heads").i64();
  c.num_key_value_heads    = tc.at("num_key_value_heads").i64();
  c.head_dim               = tc.at("head_dim").i64();
  c.vocab_size             = tc.at("vocab_size").i64();
  c.partial_rotary_factor  = tc.at("partial_rotary_factor").f64(1.0);
  c.rms_norm_eps           = tc.at("rms_norm_eps").f64(1e-6);
  c.full_attention_interval= tc.at("full_attention_interval").i64();
  c.linear_num_key_heads   = tc.at("linear_num_key_heads").i64();
  c.linear_num_value_heads = tc.at("linear_num_value_heads").i64();
  c.linear_key_head_dim    = tc.at("linear_key_head_dim").i64();
  c.linear_value_head_dim  = tc.at("linear_value_head_dim").i64();
  c.linear_conv_kernel_dim = tc.at("linear_conv_kernel_dim").i64();
  c.num_nextn_predict_layers = tc.at("num_nextn_predict_layers").i64(0);
  c.tie_word_embeddings    = root.at("tie_word_embeddings").boolean(false);

  for (const Json& lt : tc.at("layer_types").arr) {
    std::string s = lt.s();
    c.layer_types.push_back(s.find("linear") != std::string::npos ? LayerKind::LinearAttn
                                                                  : LayerKind::FullAttn);
  }

  if (root.has("quantization_config")) {
    const Json& qc = root.at("quantization_config");
    c.quant_format = qc.at("format").s();
    c.quant_method = qc.at("quant_method").s();
    const Json& g0w = qc.at("config_groups").at("group_0").at("weights");
    c.quant_group_size = g0w.at("group_size").i64();
    c.quant_num_bits   = g0w.at("num_bits").i64();
  }
  return c;
}

// Validate one NVFP4 compressed-tensors linear rooted at `prefix` (e.g.
// "...mlp.gate_proj."). Expects packed nibbles + fp8 group scale + fp32 globals.
QuantLinear Qwen35Model::check_quant_linear(const std::string& prefix) {
  QuantLinear q;
  q.packed              = prefix + "weight_packed";
  q.weight_scale        = prefix + "weight_scale";
  q.weight_global_scale = prefix + "weight_global_scale";
  q.input_global_scale  = prefix + "input_global_scale";

  auto need = [&](const std::string& n) -> bool {
    if (!st.has(n)) { report.errors.push_back("missing quant tensor: " + n); return false; }
    return true;
  };
  if (!need(q.packed) || !need(q.weight_scale) || !need(q.weight_global_scale))
    return q;

  const TensorInfo& pk = st.info(q.packed);
  const TensorInfo& ws = st.info(q.weight_scale);
  if (pk.dtype != "U8")
    report.errors.push_back(q.packed + ": expected U8, got " + pk.dtype);
  if (ws.dtype != "F8_E4M3")
    report.errors.push_back(q.weight_scale + ": expected F8_E4M3, got " + ws.dtype);
  if (pk.shape.size() != 2 || ws.shape.size() != 2) {
    report.errors.push_back(q.packed + ": rank != 2");
    return q;
  }
  q.out = pk.shape[0];
  q.in  = pk.shape[1] * 2;  // two fp4 nibbles per byte

  if (ws.shape[0] != pk.shape[0])
    report.errors.push_back(q.weight_scale + ": rows " + std::to_string(ws.shape[0]) +
                            " != packed rows " + std::to_string(pk.shape[0]));
  if (cfg.quant_group_size > 0 && ws.shape[1] * cfg.quant_group_size != q.in)
    report.errors.push_back(q.weight_scale + ": cols*group " +
                            std::to_string(ws.shape[1] * cfg.quant_group_size) +
                            " != in " + std::to_string(q.in));
  q.ok = true;
  report.n_quant_groups++;
  quant_linears_.push_back(q);
  return q;
}

void Qwen35Model::validate() {
  const auto& T = st.tensors();
  const std::string LM = "model.language_model.";

  // 1. Per-layer structure + type invariant.
  for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    std::string pfx = LM + "layers." + std::to_string(i) + ".";
    bool has_linear = false, has_full = false;
    std::set<std::string> sublinears;  // prefixes ending in weight_packed
    for (const auto& kv : T) {
      const std::string& n = kv.first;
      if (n.compare(0, pfx.size(), pfx) != 0) continue;
      if (n.find(".linear_attn.") != std::string::npos) has_linear = true;
      if (n.find(".self_attn.") != std::string::npos) has_full = true;
      const std::string suf = ".weight_packed";
      if (n.size() > suf.size() && n.compare(n.size() - suf.size(), suf.size(), suf) == 0)
        sublinears.insert(n.substr(0, n.size() - std::string("weight_packed").size()));
    }
    LayerKind want = (i < (int64_t)cfg.layer_types.size()) ? cfg.layer_types[i] : LayerKind::FullAttn;
    if (want == LayerKind::LinearAttn) {
      report.n_linear_layers++;
      if (!has_linear) report.errors.push_back("layer " + std::to_string(i) +
                        ": config says linear_attention but no linear_attn.* tensors");
      if (has_full) report.errors.push_back("layer " + std::to_string(i) +
                     ": config says linear_attention but has self_attn.* tensors");
    } else {
      report.n_full_layers++;
      if (!has_full) report.errors.push_back("layer " + std::to_string(i) +
                     ": config says full_attention but no self_attn.* tensors");
      if (has_linear) report.errors.push_back("layer " + std::to_string(i) +
                      ": config says full_attention but has linear_attn.* tensors");
    }
    for (const std::string& s : sublinears) check_quant_linear(s);
  }

  // 2. MTP head (this is what makes it "Qwen3.6-27B-MTP").
  const char* mtp_req[] = {
    "mtp.fc.weight", "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight", "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
  };
  int mtp_ok = 0;
  for (const char* n : mtp_req) if (st.has(n)) mtp_ok++;
  for (const auto& kv : T) if (kv.first.compare(0, 4, "mtp.") == 0) report.mtp_tensors++;
  report.mtp_present = (mtp_ok == (int)(sizeof(mtp_req) / sizeof(mtp_req[0])));
  if (!report.mtp_present)
    report.errors.push_back("MTP head incomplete: " + std::to_string(mtp_ok) + "/" +
                            std::to_string(sizeof(mtp_req) / sizeof(mtp_req[0])) + " required tensors");

  // 3. Embeddings / head / final norm.
  auto req_shape = [&](const std::string& n, std::vector<int64_t> want) {
    if (!st.has(n)) { report.errors.push_back("missing: " + n); return; }
    if (st.info(n).shape != want)
      report.errors.push_back(n + ": unexpected shape");
  };
  req_shape(LM + "embed_tokens.weight", {cfg.vocab_size, cfg.hidden_size});
  req_shape(LM + "norm.weight", {cfg.hidden_size});
  if (!cfg.tie_word_embeddings) req_shape("lm_head.weight", {cfg.vocab_size, cfg.hidden_size});

  // 4. Vision tower present (kept BF16, in quant ignore list).
  for (const auto& kv : T)
    if (kv.first.compare(0, 13, "model.visual.") == 0) { report.vision_present = true; break; }

  // 5. Global stats.
  for (const auto& kv : T) {
    report.n_tensors_seen++;
    report.dtype_counts[kv.second.dtype]++;
    report.dtype_bytes[kv.second.dtype] += kv.second.nbytes();
    report.bytes_total += kv.second.nbytes();
  }

  report.pass = report.errors.empty() && report.mtp_present &&
                report.n_linear_layers + report.n_full_layers == cfg.num_hidden_layers;
}

void Qwen35Model::load(const std::string& dir) {
  cfg = Qwen35Config::from_file(dir + "/config.json");
  st.open(dir + "/model.safetensors");
  quant_linears_.clear();
  validate();  // populates quant_linears_ (one entry per validated NVFP4 linear)
}

const QuantLinear* Qwen35Model::first_quant_linear() const {
  return quant_linears_.empty() ? nullptr : &quant_linears_.front();
}

void Qwen35Model::print_summary() const {
  printf("== Qwen3.6-27B (Qwen3_5ForConditionalGeneration) ==\n");
  printf("  layers          : %ld  (%ld linear-attn + %ld full-attn, config interval=%ld)\n",
         (long)cfg.num_hidden_layers, (long)report.n_linear_layers,
         (long)report.n_full_layers, (long)cfg.full_attention_interval);
  printf("  hidden/heads/kv : %ld / %ld / %ld   head_dim=%ld  vocab=%ld\n",
         (long)cfg.hidden_size, (long)cfg.num_attention_heads,
         (long)cfg.num_key_value_heads, (long)cfg.head_dim, (long)cfg.vocab_size);
  printf("  GDN             : key_heads=%ld val_heads=%ld kdim=%ld vdim=%ld conv=%ld\n",
         (long)cfg.linear_num_key_heads, (long)cfg.linear_num_value_heads,
         (long)cfg.linear_key_head_dim, (long)cfg.linear_value_head_dim,
         (long)cfg.linear_conv_kernel_dim);
  printf("  quant           : %s / %s  (%ld-bit, group=%ld)\n",
         cfg.quant_method.c_str(), cfg.quant_format.c_str(),
         (long)cfg.quant_num_bits, (long)cfg.quant_group_size);
  printf("  NVFP4 linears   : %ld validated\n", (long)report.n_quant_groups);
  printf("  MTP head        : %s  (%ld tensors)\n",
         report.mtp_present ? "PRESENT" : "MISSING", (long)report.mtp_tensors);
  printf("  vision tower    : %s\n", report.vision_present ? "present (bf16)" : "absent");
  printf("  tensors         : %ld  (%.2f GB mmapped)\n",
         (long)report.n_tensors_seen, report.bytes_total / 1e9);
  printf("  dtypes          :");
  for (const auto& kv : report.dtype_counts)
    printf(" %s=%d(%.1fG)", kv.first.c_str(), kv.second, report.dtype_bytes.at(kv.first) / 1e9);
  printf("\n");
  if (!report.warnings.empty()) {
    printf("  warnings        : %zu\n", report.warnings.size());
    for (auto& w : report.warnings) printf("    ! %s\n", w.c_str());
  }
  if (!report.errors.empty()) {
    printf("  ERRORS          : %zu\n", report.errors.size());
    for (size_t i = 0; i < report.errors.size() && i < 20; ++i)
      printf("    x %s\n", report.errors[i].c_str());
  }
}

}  // namespace lb
