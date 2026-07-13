// Verify the byte-level BPE tokenizer + chat template against a golden battery
// generated from the HF tokenizer (test/tok_battery.json).
//   ./lbtest_tok <model_dir> [battery.json]
#include "tokenizer.hpp"
#include "json.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using namespace lb;

static std::string slurp(const std::string& p) {
  std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static std::vector<int> ints(const Json& a) {
  std::vector<int> v; for (auto& x : a.arr) v.push_back((int)x.i64()); return v;
}
static std::string show(const std::vector<int>& v, int lim = 24) {
  std::string s; for (int i = 0; i < (int)v.size() && i < lim; ++i) s += std::to_string(v[i]) + " ";
  return s;
}

int main(int argc, char** argv) {
  std::string mdir = argc > 1 ? argv[1] : "/model";
  std::string bat = argc > 2 ? argv[2] : "test/tok_battery.json";
  Tokenizer tok; tok.load(mdir);
  printf("loaded tokenizer: vocab=%d eos=%d\n", tok.vocab_size(), tok.eos());

  Json root = json_parse(slurp(bat));
  int pass = 0, fail = 0;
  for (auto& cj : root.at("cases").arr) {
    std::string s = cj.at("s").s();
    std::vector<int> exp = ints(cj.at("ids"));
    std::vector<int> got = tok.encode(s, true);
    bool ok = got == exp;
    // roundtrip decode
    std::string dec = tok.decode(got, true);
    bool rt = (dec == s);
    if (ok) ++pass; else { ++fail;
      printf("FAIL %-32s\n  exp: %s\n  got: %s\n", ("\"" + s.substr(0, 30) + "\"").c_str(),
             show(exp).c_str(), show(got).c_str());
    }
    if (!rt && ok) printf("  (decode roundtrip differs for %.20s: %.30s)\n", s.c_str(), dec.c_str());
  }
  printf("encode: %d/%d pass\n", pass, pass + fail);

  // chat template
  std::vector<ChatMsg> msgs = {{"system", "You are helpful."}, {"user", "Hi there"}};
  std::string rendered = tok.apply_chat_template(msgs, true, true);
  std::string exp_text = root.at("chat").at("text").s();
  std::vector<int> chat_exp = ints(root.at("chat").at("ids"));
  std::vector<int> chat_got = tok.encode(rendered, true);
  bool tpl_ok = rendered == exp_text;
  bool cids_ok = chat_got == chat_exp;
  printf("chat template render: %s\n", tpl_ok ? "PASS" : "FAIL");
  if (!tpl_ok) printf("  exp: %s\n  got: %s\n", exp_text.c_str(), rendered.c_str());
  printf("chat token ids:       %s\n", cids_ok ? "PASS" : "FAIL");
  if (!cids_ok) printf("  exp: %s\n  got: %s\n", show(chat_exp).c_str(), show(chat_got).c_str());

  bool all = (fail == 0) && tpl_ok && cids_ok;
  printf("== %s ==\n", all ? "PASS" : "FAIL");
  return all ? 0 : 1;
}
