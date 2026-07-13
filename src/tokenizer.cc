#include "tokenizer.hpp"
#include "unicode.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include <stdexcept>

namespace lb {

static std::string read_file(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::string s(n, '\0');
  size_t got = fread(&s[0], 1, n, f); fclose(f);
  s.resize(got);
  return s;
}

// Parse a JSON string literal at content[i]=='"'; return unescaped bytes, advance i past close.
static std::string json_str(const std::string& c, size_t& i) {
  std::string out;
  ++i;  // opening quote
  while (i < c.size() && c[i] != '"') {
    char ch = c[i++];
    if (ch == '\\' && i < c.size()) {
      char e = c[i++];
      switch (e) {
        case 'n': out += '\n'; break; case 't': out += '\t'; break;
        case 'r': out += '\r'; break; case 'b': out += '\b'; break;
        case 'f': out += '\f'; break; case '/': out += '/'; break;
        case '\\': out += '\\'; break; case '"': out += '"'; break;
        case 'u': {
          unsigned v = (unsigned)strtoul(c.substr(i, 4).c_str(), nullptr, 16); i += 4;
          if (v >= 0xD800 && v <= 0xDBFF && i + 6 <= c.size() && c[i] == '\\' && c[i + 1] == 'u') {
            unsigned lo = (unsigned)strtoul(c.substr(i + 2, 4).c_str(), nullptr, 16); i += 6;
            v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8_append(out, v); break;
        }
        default: out += e;
      }
    } else out += ch;
  }
  ++i;  // closing quote
  return out;
}

void Tokenizer::load(const std::string& dir) {
  // byte-level alphabet (GPT-2 bytes_to_unicode)
  std::vector<int> bs;
  for (int b = 33; b <= 126; ++b) bs.push_back(b);
  for (int b = 161; b <= 172; ++b) bs.push_back(b);
  for (int b = 174; b <= 255; ++b) bs.push_back(b);
  std::vector<int> cs = bs;
  int nextcp = 256, n = 0;
  std::vector<int> b2c(256, -1);
  for (int b : bs) b2c[b] = b;
  for (int b = 0; b < 256; ++b) if (b2c[b] < 0) { b2c[b] = nextcp++; (void)n; }
  for (int b = 0; b < 256; ++b) {
    std::string u; utf8_append(u, (uint32_t)b2c[b]);
    byte2uni_[b] = u; uni2byte_[u] = b;
  }

  std::string c = read_file(dir + "/tokenizer.json");

  // Locate a JSON key whose value opens with `open` ({ or [). Skips homonym
  // tokens like "vocab": 83849 (a BPE token) to find the real container.
  auto find_container = [&](const char* key, char open) -> size_t {
    size_t klen = strlen(key), p = 0;
    while ((p = c.find(key, p)) != std::string::npos) {
      size_t q = p + klen;
      while (q < c.size() && (c[q] == ' ' || c[q] == '\t' || c[q] == '\n' || c[q] == '\r')) ++q;
      if (q < c.size() && c[q] == open) return q + 1;   // just past the open bracket
      p += klen;
    }
    return std::string::npos;
  };

  // --- vocab: "vocab":{ "tok":id, ... } ---
  size_t vp = find_container("\"vocab\":", '{');
  if (vp == std::string::npos) throw std::runtime_error("tokenizer: no vocab");
  int max_id = 0;
  std::vector<std::pair<std::string, int>> pairs;
  while (vp < c.size()) {
    while (vp < c.size() && (c[vp] == ' ' || c[vp] == '\n' || c[vp] == ',' || c[vp] == '\r' || c[vp] == '\t')) ++vp;
    if (c[vp] == '}') { ++vp; break; }
    std::string tok = json_str(c, vp);
    while (vp < c.size() && c[vp] != ':') ++vp;
    ++vp;
    while (vp < c.size() && c[vp] == ' ') ++vp;
    int id = (int)strtol(c.c_str() + vp, nullptr, 10);
    while (vp < c.size() && (isdigit(c[vp]) || c[vp] == '-')) ++vp;
    pairs.push_back({tok, id});
    if (id > max_id) max_id = id;
  }

  // --- added_tokens (specials) --- parse before sizing id2tok_
  size_t ap = find_container("\"added_tokens\":", '[');
  if (ap != std::string::npos) {
    size_t aend = c.find("\"normalizer\"", ap);
    if (aend == std::string::npos) aend = c.size();
    size_t i = ap;
    while (i < aend) {
      size_t idp = c.find("\"id\":", i);
      if (idp == std::string::npos || idp >= aend) break;
      size_t q = idp + 5; while (q < c.size() && c[q] == ' ') ++q;
      int id = (int)strtol(c.c_str() + q, nullptr, 10);
      size_t cp = c.find("\"content\":", idp);
      size_t cq = c.find('"', cp + 10);
      std::string content = json_str(c, cq);
      specials_.push_back({content, id});
      special_by_id_[id] = content;
      if (id > max_id) max_id = id;
      i = cq;
    }
  }
  std::sort(specials_.begin(), specials_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  id2tok_.assign(max_id + 1, std::string());
  for (auto& kv : pairs) { vocab_[kv.first] = kv.second; id2tok_[kv.second] = kv.first; }

  // --- merges: "merges":[ ["a","b"], ... ] ---
  size_t mp = find_container("\"merges\":", '[');
  int rank = 0;
  while (mp < c.size()) {
    while (mp < c.size() && c[mp] != '[' && c[mp] != ']') ++mp;
    if (c[mp] == ']') break;
    ++mp;                                   // into pair
    while (c[mp] != '"') ++mp;
    std::string a = json_str(c, mp);
    while (c[mp] != '"') ++mp;
    std::string b = json_str(c, mp);
    while (c[mp] != ']') ++mp;
    ++mp;                                   // past pair's ]
    merge_rank_[a + " " + b] = rank++;
  }

  eos_id_ = im_end_ = -1;
  for (auto& s : specials_) {
    if (s.first == "<|im_start|>") im_start_ = s.second;
    if (s.first == "<|im_end|>") { im_end_ = s.second; eos_id_ = s.second; }
  }
}

// GPT-4-style pre-tokenize: (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   | \p{N} |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
std::vector<std::pair<size_t, size_t>> Tokenizer::pretokenize(const std::vector<uint32_t>& cp) const {
  std::vector<std::pair<size_t, size_t>> out;
  size_t i = 0, n = cp.size();
  auto lower = [](uint32_t x) { return (x >= 'A' && x <= 'Z') ? x + 32 : x; };
  while (i < n) {
    // Alt1: contractions
    if (cp[i] == '\'' && i + 1 < n) {
      uint32_t a = lower(cp[i + 1]);
      uint32_t b = i + 2 < n ? lower(cp[i + 2]) : 0;
      int len = 0;
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) len = 2;
      else if (a == 's' || a == 't' || a == 'm' || a == 'd') len = 1;
      if (len) { out.push_back({i, i + 1 + len}); i += 1 + len; continue; }
    }
    // Alt2: optional non-(nl|L|N) prefix + (L|M)+
    {
      size_t j = i;
      bool pfx = !is_nl(cp[i]) && !is_letter(cp[i]) && !is_number(cp[i]);
      if (pfx && i + 1 < n && (is_letter(cp[i + 1]) || is_mark(cp[i + 1]))) j = i + 1;
      else if (pfx) j = i;  // prefix only valid if letters follow; else fall through below
      if (j < n && (is_letter(cp[j]) || is_mark(cp[j]))) {
        while (j < n && (is_letter(cp[j]) || is_mark(cp[j]))) ++j;
        out.push_back({i, j}); i = j; continue;
      }
    }
    // Alt3: single number
    if (is_number(cp[i])) { out.push_back({i, i + 1}); ++i; continue; }
    // Alt4: optional space + symbols + trailing newlines
    {
      size_t j = i;
      auto sym = [&](uint32_t x) { return !is_ws(x) && !is_letter(x) && !is_number(x) && !is_mark(x); };
      if (cp[i] == 0x20 && i + 1 < n && sym(cp[i + 1])) j = i + 1;
      size_t k = j;
      while (k < n && sym(cp[k])) ++k;
      if (k > j) {
        while (k < n && is_nl(cp[k])) ++k;
        out.push_back({i, k}); i = k; continue;
      }
    }
    // Alt5: whitespace run ending in a newline
    if (is_ws(cp[i])) {
      size_t j = i; long last_nl = -1;
      while (j < n && is_ws(cp[j])) { if (is_nl(cp[j])) last_nl = (long)j; ++j; }
      if (last_nl >= 0) { out.push_back({i, (size_t)last_nl + 1}); i = (size_t)last_nl + 1; continue; }
      // Alt6/7: trailing vs leading whitespace
      if (j == n) { out.push_back({i, j}); i = j; }
      else if (j - 1 > i) { out.push_back({i, j - 1}); i = j - 1; }  // leave 1 ws for next word
      else { out.push_back({i, j}); i = j; }
      continue;
    }
    out.push_back({i, i + 1}); ++i;  // fallback
  }
  return out;
}

static std::vector<std::string> split_chars(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    uint8_t ch = s[i]; int len = ch < 0x80 ? 1 : (ch >> 5) == 0x6 ? 2 : (ch >> 4) == 0xE ? 3 : 4;
    out.push_back(s.substr(i, len)); i += len;
  }
  return out;
}

void Tokenizer::bpe(const std::string& piece, std::vector<int>& out) const {
  std::vector<std::string> sym = split_chars(piece);
  while (sym.size() >= 2) {
    int best = INT_MAX; size_t bi = 0;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
      if (it != merge_rank_.end() && it->second < best) { best = it->second; bi = i; }
    }
    if (best == INT_MAX) break;
    sym[bi] += sym[bi + 1];
    sym.erase(sym.begin() + bi + 1);
  }
  for (auto& s : sym) {
    auto it = vocab_.find(s);
    if (it != vocab_.end()) out.push_back(it->second);
    else for (char ch : s) { auto j = vocab_.find(byte2uni_[(uint8_t)ch]); if (j != vocab_.end()) out.push_back(j->second); }
  }
}

void Tokenizer::encode_ordinary(const std::string& text, std::vector<int>& out) const {
  if (text.empty()) return;
  std::vector<uint32_t> cps = utf8_decode(text);
  for (auto& r : pretokenize(cps)) {
    std::string bytes, piece;
    for (size_t k = r.first; k < r.second; ++k) utf8_append(bytes, cps[k]);   // pretoken -> utf8 bytes
    for (unsigned char ch : bytes) piece += byte2uni_[ch];                    // bytes -> byte-level chars
    bpe(piece, out);
  }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_special) const {
  std::vector<int> out;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t best = std::string::npos; size_t blen = 0; int bid = -1;
    if (add_special) {
      for (auto& s : specials_) {
        size_t f = text.find(s.first, pos);
        if (f != std::string::npos && (f < best || (f == best && s.first.size() > blen))) {
          best = f; blen = s.first.size(); bid = s.second;
        }
      }
    }
    if (bid >= 0) {
      encode_ordinary(text.substr(pos, best - pos), out);
      out.push_back(bid);
      pos = best + blen;
    } else { encode_ordinary(text.substr(pos), out); break; }
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
  std::string out, bytebuf;
  auto flush = [&]() {
    for (auto& ch : split_chars(bytebuf)) { auto it = uni2byte_.find(ch); if (it != uni2byte_.end()) out += (char)it->second; }
    bytebuf.clear();
  };
  for (int id : ids) {
    if (is_special(id)) { flush(); if (!skip_special) out += special_by_id_.at(id); }
    else if (id >= 0 && id < (int)id2tok_.size()) bytebuf += id2tok_[id];
  }
  flush();
  return out;
}

std::string Tokenizer::apply_chat_template(const std::vector<ChatMsg>& msgs, bool add_gen_prompt,
                                           bool enable_thinking) const {
  std::string s;
  for (auto& m : msgs) s += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
  if (add_gen_prompt) {
    s += "<|im_start|>assistant\n";
    s += enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  return s;
}

}  // namespace lb
