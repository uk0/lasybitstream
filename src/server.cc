// OpenAI-compatible HTTP server for the native Qwen3.6-27B engine.
//   POST /v1/chat/completions   (stream + non-stream)
//   GET  /v1/models
//   GET  /health
// Pure C++ + BSD sockets; binds 127.0.0.1 only.
//   ./lbserve <model_dir> [port] [max_tokens]
#include "engine.hpp"
#include "tokenizer.hpp"
#include "json.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <ctime>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

using namespace lb;

static Engine* g_eng = nullptr;
static Tokenizer* g_tok = nullptr;
static std::string g_model = "qwen3.6-27b";
static int g_max_tokens = 256;

static std::string json_escape(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break; case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break; case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break; case '\b': o += "\\b"; break; case '\f': o += "\\f"; break;
      default: if (c < 0x20) { char b[8]; snprintf(b, 8, "\\u%04x", c); o += b; } else o += (char)c;
    }
  }
  return o;
}

// Extract the text of a chat message: content may be a string or an array of
// {type:"text"|"image_url", ...} parts (OpenAI multimodal format).
static std::string content_text(const Json& content, bool& had_image) {
  if (content.is_str()) return content.s();
  std::string out;
  if (content.is_arr())
    for (auto& part : content.arr) {
      std::string ty = part.at("type").s();
      if (ty == "text") out += part.at("text").s();
      else if (ty == "image_url") had_image = true;   // vision tower not yet wired
    }
  return out;
}

static void send_all(int fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) { ssize_t n = send(fd, s.data() + off, s.size() - off, 0); if (n <= 0) break; off += n; }
}

static std::string http_json(const std::string& body, int code = 200) {
  std::string status = code == 200 ? "200 OK" : code == 400 ? "400 Bad Request" : "404 Not Found";
  return "HTTP/1.1 " + status + "\r\nContent-Type: application/json\r\n"
         "Access-Control-Allow-Origin: *\r\nConnection: close\r\nContent-Length: " +
         std::to_string(body.size()) + "\r\n\r\n" + body;
}

static long now_s() { return (long)time(nullptr); }

static void handle_chat(int fd, const Json& req) {
  // build the ChatML prompt from messages
  std::vector<ChatMsg> msgs;
  bool had_image = false, enable_thinking = true;
  if (req.has("enable_thinking")) enable_thinking = req.at("enable_thinking").boolean(true);
  for (auto& m : req.at("messages").arr)
    msgs.push_back({m.at("role").s(), content_text(m.at("content"), had_image)});
  int max_new = req.has("max_tokens") ? (int)req.at("max_tokens").i64(g_max_tokens) : g_max_tokens;
  if (max_new <= 0 || max_new > 2048) max_new = g_max_tokens;
  bool stream = req.has("stream") && req.at("stream").boolean(false);

  std::string prompt = g_tok->apply_chat_template(msgs, true, enable_thinking);
  std::vector<int> ids = g_tok->encode(prompt, true);
  int prompt_tok = (int)ids.size();
  int eos = g_tok->eos();
  char idbuf[64]; snprintf(idbuf, 64, "chatcmpl-%ld", now_s());

  if (!stream) {
    std::vector<int> out = g_eng->generate(ids, max_new, eos);
    std::string text = g_tok->decode(out, true);
    std::string body = std::string("{\"id\":\"") + idbuf + "\",\"object\":\"chat.completion\",\"created\":" +
      std::to_string(now_s()) + ",\"model\":\"" + g_model + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
      json_escape(text) + "\"},\"finish_reason\":\"" + ((int)out.size() >= max_new ? "length" : "stop") + "\"}],\"usage\":{\"prompt_tokens\":" +
      std::to_string(prompt_tok) + ",\"completion_tokens\":" + std::to_string((int)out.size()) +
      ",\"total_tokens\":" + std::to_string(prompt_tok + (int)out.size()) + "}}";
    send_all(fd, http_json(body));
    return;
  }

  // streaming (SSE): one chunk per decoded UTF-8-complete delta
  std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                     "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
  send_all(fd, head);
  auto sse = [&](const std::string& delta, const char* finish) {
    std::string d = std::string("{\"id\":\"") + idbuf + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
      std::to_string(now_s()) + ",\"model\":\"" + g_model + "\",\"choices\":[{\"index\":0,\"delta\":" +
      (delta.empty() ? std::string("{}") : std::string("{\"content\":\"") + json_escape(delta) + "\"}") +
      ",\"finish_reason\":" + (finish ? std::string("\"") + finish + "\"" : std::string("null")) + "}]}";
    send_all(fd, "data: " + d + "\n\n");
  };
  sse("", nullptr);  // role chunk (delta {} keeps it simple)
  std::vector<int> acc;
  std::string sent;
  // Largest prefix length of s that ends on a complete UTF-8 character.
  auto safe_end = [](const std::string& s) -> size_t {
    if (s.empty()) return 0;
    size_t p = s.size();
    while (p > 0 && ((unsigned char)s[p - 1] & 0xC0) == 0x80) --p;   // skip continuation bytes
    if (p == 0) return s.size();
    --p;                                                            // last lead byte
    unsigned char lead = s[p];
    int need = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : 4;
    return (p + (size_t)need <= s.size()) ? s.size() : p;           // complete -> all; else cut before it
  };
  g_eng->generate(ids, max_new, eos, [&](int tid) {
    acc.push_back(tid);
    std::string full = g_tok->decode(acc, true);
    size_t end = safe_end(full);
    if (end > sent.size()) { sse(full.substr(sent.size(), end - sent.size()), nullptr); sent = full.substr(0, end); }
  });
  std::string full = g_tok->decode(acc, true);
  if (full.size() > sent.size()) sse(full.substr(sent.size()), nullptr);
  sse("", (int)acc.size() >= max_new ? "length" : "stop");
  send_all(fd, "data: [DONE]\n\n");
}

static void handle(int fd) {
  std::string req;
  char buf[8192];
  ssize_t n;
  // read headers
  size_t hdr_end;
  while ((hdr_end = req.find("\r\n\r\n")) == std::string::npos) {
    n = recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) return;
    req.append(buf, n);
    if (req.size() > (1 << 20)) break;
  }
  size_t clen = 0;
  { size_t p = req.find("Content-Length:");
    if (p == std::string::npos) p = req.find("content-length:");
    if (p != std::string::npos) clen = strtoul(req.c_str() + p + 15, nullptr, 10); }
  size_t body_start = hdr_end + 4;
  while (req.size() - body_start < clen) {
    n = recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) break;
    req.append(buf, n);
  }
  std::string method = req.substr(0, req.find(' '));
  size_t ps = req.find(' ') + 1;
  std::string path = req.substr(ps, req.find(' ', ps) - ps);

  if (method == "GET" && path == "/health") { send_all(fd, http_json("{\"status\":\"ok\"}")); return; }
  if (method == "GET" && path == "/v1/models") {
    send_all(fd, http_json("{\"object\":\"list\",\"data\":[{\"id\":\"" + g_model +
                           "\",\"object\":\"model\",\"owned_by\":\"lasybitstream\"}]}"));
    return;
  }
  if (method == "POST" && path == "/v1/chat/completions") {
    try {
      Json r = json_parse(req.data() + body_start, clen);
      handle_chat(fd, r);
    } catch (std::exception& e) {
      send_all(fd, http_json(std::string("{\"error\":{\"message\":\"") + json_escape(e.what()) + "\"}}", 400));
    }
    return;
  }
  send_all(fd, http_json("{\"error\":{\"message\":\"not found\"}}", 404));
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: %s <model_dir> [port] [max_tokens]\n", argv[0]); return 1; }
  std::string mdir = argv[1];
  int port = argc > 2 ? atoi(argv[2]) : 8080;
  if (argc > 3) g_max_tokens = atoi(argv[3]);

  Tokenizer tok; tok.load(mdir);
  Engine eng; eng.load(mdir, 4096);
  g_tok = &tok; g_eng = &eng;
  printf("loaded engine + tokenizer (vocab=%d, eos=%d)\n", tok.vocab_size(), tok.eos());

  int s = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(port);
  addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  if (bind(s, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
  listen(s, 16);
  printf("lasybitstream OpenAI server on http://127.0.0.1:%d  (POST /v1/chat/completions)\n", port);
  fflush(stdout);
  while (true) {
    int fd = accept(s, nullptr, nullptr);
    if (fd < 0) continue;
    handle(fd);
    close(fd);
  }
  return 0;
}
