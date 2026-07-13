// Byte-level BPE tokenizer for Qwen3.6 — loads tokenizer.json directly, applies
// the GPT-4-style pre-tokenize regex, merges by rank, and handles the ChatML
// chat template (<|im_start|>… + <think>). Pure C++, zero deps.
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace lb {

struct ChatMsg { std::string role, content; };

class Tokenizer {
 public:
  void load(const std::string& model_dir);
  std::vector<int> encode(const std::string& text, bool add_special = true) const;
  std::string decode(const std::vector<int>& ids, bool skip_special = true) const;
  // ChatML render matching apply_chat_template(add_generation_prompt=True).
  std::string apply_chat_template(const std::vector<ChatMsg>& msgs,
                                  bool add_gen_prompt = true, bool enable_thinking = true) const;
  int eos() const { return eos_id_; }
  int im_end() const { return im_end_; }
  int vocab_size() const { return (int)id2tok_.size(); }
  bool is_special(int id) const { return special_by_id_.count(id) != 0; }

 private:
  std::unordered_map<std::string, int> vocab_;         // byte-level token -> id
  std::vector<std::string> id2tok_;                    // id -> byte-level token
  std::unordered_map<std::string, int> merge_rank_;    // "a b" -> rank
  std::vector<std::pair<std::string, int>> specials_;  // content -> id (len desc)
  std::unordered_map<int, std::string> special_by_id_;
  std::string byte2uni_[256];                          // byte -> utf8 of byte-level char
  std::unordered_map<std::string, int> uni2byte_;      // byte-level char -> byte
  int eos_id_ = -1, im_start_ = -1, im_end_ = -1;

  std::vector<std::pair<size_t, size_t>> pretokenize(const std::vector<uint32_t>& cps) const;
  void bpe(const std::string& piece, std::vector<int>& out) const;
  void encode_ordinary(const std::string& text, std::vector<int>& out) const;
};

}  // namespace lb
