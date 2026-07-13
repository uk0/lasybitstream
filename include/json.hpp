// Minimal zero-dependency JSON parser — enough for HF config.json and the
// safetensors header (objects, arrays, strings, numbers, bool, null).
#pragma once
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <cstdlib>

namespace lb {

struct Json {
  enum Type { Null, Bool, Num, Str, Arr, Obj };
  Type t = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  bool is_obj() const { return t == Obj; }
  bool is_arr() const { return t == Arr; }
  bool is_str() const { return t == Str; }
  bool is_num() const { return t == Num; }
  bool has(const std::string& k) const { return t == Obj && obj.count(k); }
  const Json& at(const std::string& k) const {
    static Json nul;
    auto it = obj.find(k);
    return it == obj.end() ? nul : it->second;
  }
  const Json& operator[](const std::string& k) const { return at(k); }
  long i64(long d = 0) const { return t == Num ? (long)num : d; }
  double f64(double d = 0) const { return t == Num ? num : d; }
  std::string s(const std::string& d = "") const { return t == Str ? str : d; }
  bool boolean(bool d = false) const { return t == Bool ? b : d; }
};

class JsonParser {
  const char* p;
  const char* end;
  void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
  [[noreturn]] void err(const char* m) { throw std::runtime_error(std::string("json: ") + m); }

  std::string parse_str() {
    if (*p != '"') err("expected string");
    ++p;
    std::string out;
    while (p < end && *p != '"') {
      char c = *p++;
      if (c == '\\') {
        if (p >= end) err("bad escape");
        char e = *p++;
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case '/': out += '/'; break;
          case '\\': out += '\\'; break;
          case '"': out += '"'; break;
          case 'u': {
            if (p + 4 > end) err("bad \\u");
            unsigned v = (unsigned)strtoul(std::string(p, p + 4).c_str(), nullptr, 16);
            p += 4;
            if (v < 0x80) out += (char)v;
            else if (v < 0x800) { out += (char)(0xC0 | (v >> 6)); out += (char)(0x80 | (v & 0x3F)); }
            else { out += (char)(0xE0 | (v >> 12)); out += (char)(0x80 | ((v >> 6) & 0x3F)); out += (char)(0x80 | (v & 0x3F)); }
            break;
          }
          default: out += e;
        }
      } else out += c;
    }
    if (p >= end) err("unterminated string");
    ++p;  // closing quote
    return out;
  }

  Json parse_val() {
    ws();
    if (p >= end) err("unexpected end");
    char c = *p;
    Json j;
    if (c == '"') { j.t = Json::Str; j.str = parse_str(); }
    else if (c == '{') {
      j.t = Json::Obj; ++p; ws();
      if (*p == '}') { ++p; return j; }
      while (true) {
        ws();
        std::string k = parse_str();
        ws();
        if (*p != ':') err("expected ':'");
        ++p;
        j.obj[k] = parse_val();
        ws();
        if (*p == ',') { ++p; continue; }
        if (*p == '}') { ++p; break; }
        err("expected ',' or '}'");
      }
    }
    else if (c == '[') {
      j.t = Json::Arr; ++p; ws();
      if (*p == ']') { ++p; return j; }
      while (true) {
        j.arr.push_back(parse_val());
        ws();
        if (*p == ',') { ++p; continue; }
        if (*p == ']') { ++p; break; }
        err("expected ',' or ']'");
      }
    }
    else if (c == 't') { if (end - p < 4) err("bad true"); j.t = Json::Bool; j.b = true; p += 4; }
    else if (c == 'f') { if (end - p < 5) err("bad false"); j.t = Json::Bool; j.b = false; p += 5; }
    else if (c == 'n') { if (end - p < 4) err("bad null"); j.t = Json::Null; p += 4; }
    else {  // number
      char* np;
      j.t = Json::Num;
      j.num = strtod(p, &np);
      if (np == p) err("bad number");
      p = np;
    }
    return j;
  }

 public:
  JsonParser(const char* data, size_t n) : p(data), end(data + n) {}
  Json parse() { Json j = parse_val(); ws(); return j; }
};

inline Json json_parse(const char* data, size_t n) { return JsonParser(data, n).parse(); }
inline Json json_parse(const std::string& s) { return JsonParser(s.data(), s.size()).parse(); }

}  // namespace lb
