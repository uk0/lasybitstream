// OpenAI-compatible HTTP server for the native Qwen3.6-27B engine.
//   POST /v1/chat/completions   (stream + non-stream)
//   GET  /v1/models
//   GET  /health
// Pure C++ + BSD sockets; binds 127.0.0.1 only.
//   ./lbserve <model_dir> [port] [max_tokens]
#include "engine.hpp"
#include "tokenizer.hpp"
#include "preprocess.hpp"
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

// Extract the text of a chat message; if a part carries an image, its source
// (a data: URL, an http(s) URL, or raw base64) is returned in `image_src`.
// Handles OpenAI {type:"image_url",image_url:{url}} and Anthropic
// {type:"image",source:{type:"base64",data} | {type:"url",url}}.
static std::string content_text(const Json& content, std::string& image_src) {
  if (content.is_str()) return content.s();
  std::string out;
  if (content.is_arr())
    for (auto& part : content.arr) {
      std::string ty = part.at("type").s();
      if (ty == "text") out += part.at("text").s();
      else if (ty == "image_url") image_src = part.at("image_url").at("url").s();
      else if (ty == "image") {
        const Json& src = part.at("source");
        image_src = src.has("data") ? src.at("data").s() : src.at("url").s();
      }
    }
  return out;
}

// Fetch decoded image bytes from a source (data: URL base64, http(s) URL via curl,
// or raw base64).
static std::vector<uint8_t> fetch_image(const std::string& src) {
  if (src.rfind("http", 0) == 0) {                     // download via curl
    std::string cmd = "curl -sL --max-time 20 '" + src + "'";
    FILE* p = popen(cmd.c_str(), "r");
    std::vector<uint8_t> b; if (!p) return b;
    uint8_t buf[65536]; size_t n;
    while ((n = fread(buf, 1, sizeof(buf), p)) > 0) b.insert(b.end(), buf, buf + n);
    pclose(p); return b;
  }
  size_t comma = src.find("base64,");                  // data: URL -> strip prefix
  std::string b64 = comma != std::string::npos ? src.substr(comma + 7) : src;
  return base64_decode(b64);
}

// Largest prefix length of s that ends on a complete UTF-8 character, so a
// multi-byte char (e.g. CJK) is never split across two streamed deltas.
static size_t utf8_safe_end(const std::string& s) {
  if (s.empty()) return 0;
  size_t p = s.size();
  while (p > 0 && ((unsigned char)s[p - 1] & 0xC0) == 0x80) --p;   // skip continuation bytes
  if (p == 0) return s.size();
  --p;                                                            // last lead byte
  unsigned char lead = s[p];
  int need = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : 4;
  return (p + (size_t)need <= s.size()) ? s.size() : p;
}

static const int IMAGE_TOKEN_ID = 248056;              // <|image_pad|>

// Build ChatMsgs from the request; if a message carries an image, fetch+preprocess+
// encode it and splice <|vision_start|>…<|image_pad|>×N…<|vision_end|> into that
// message. Returns the merged image-token count (0 = text only); sets gh/gw.
static int build_msgs(const Json& messages, std::vector<ChatMsg>& msgs, int& gh, int& gw) {
  int merged = 0;
  for (auto& m : messages.arr) {
    std::string img_src, text = content_text(m.at("content"), img_src), content = text;
    if (!img_src.empty() && merged == 0) {             // one image per request supported
      std::vector<uint8_t> bytes = fetch_image(img_src);
      if (!bytes.empty()) {
        PreImage pre = preprocess_image(bytes.data(), bytes.size());
        if (pre.ok) {
          merged = g_eng->encode_image(pre.pixels, pre.t, pre.h, pre.w);
          gh = pre.h; gw = pre.w;
          std::string pads; for (int i = 0; i < merged; ++i) pads += "<|image_pad|>";
          content = "<|vision_start|>" + pads + "<|vision_end|>" + text;
        }
      }
    }
    msgs.push_back({m.at("role").s(), content});
  }
  return merged;
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
  // build the ChatML prompt from messages (encoding an image if present)
  std::vector<ChatMsg> msgs; int gh = 0, gw = 0;
  bool enable_thinking = req.has("enable_thinking") ? req.at("enable_thinking").boolean(true) : true;
  int merged = build_msgs(req.at("messages"), msgs, gh, gw);
  int max_new = req.has("max_tokens") ? (int)req.at("max_tokens").i64(g_max_tokens) : g_max_tokens;
  if (max_new <= 0 || max_new > 2048) max_new = g_max_tokens;
  bool stream = req.has("stream") && req.at("stream").boolean(false);

  std::string prompt = g_tok->apply_chat_template(msgs, true, enable_thinking);
  std::vector<int> ids = g_tok->encode(prompt, true);
  int prompt_tok = (int)ids.size();
  int eos = g_tok->eos();
  char idbuf[64]; snprintf(idbuf, 64, "chatcmpl-%ld", now_s());

  if (!stream) {
    std::vector<int> out = merged > 0 ? g_eng->generate_mm(ids, IMAGE_TOKEN_ID, gh, gw, max_new, eos)
                                      : g_eng->generate_spec(ids, max_new, eos, 1);   // MTP = faster
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
  auto cb = [&](int tid) {
    acc.push_back(tid);
    std::string full = g_tok->decode(acc, true);
    size_t end = utf8_safe_end(full);
    if (end > sent.size()) { sse(full.substr(sent.size(), end - sent.size()), nullptr); sent = full.substr(0, end); }
  };
  if (merged > 0) g_eng->generate_mm(ids, IMAGE_TOKEN_ID, gh, gw, max_new, eos, cb);
  else g_eng->generate_spec(ids, max_new, eos, 1, cb);   // MTP speculative decode (faster single-stream)
  std::string full = g_tok->decode(acc, true);
  if (full.size() > sent.size()) sse(full.substr(sent.size()), nullptr);
  sse("", (int)acc.size() >= max_new ? "length" : "stop");
  send_all(fd, "data: [DONE]\n\n");
}

// Anthropic Messages API: POST /v1/messages. Differs from OpenAI — `system` is a
// top-level field (string or text blocks), and the SSE stream is an event sequence.
static void handle_messages(int fd, const Json& req) {
  std::vector<ChatMsg> msgs; int gh = 0, gw = 0;
  if (req.has("system")) {
    const Json& s = req.at("system");
    std::string systext;
    if (s.is_str()) systext = s.s();
    else if (s.is_arr()) for (auto& blk : s.arr) if (blk.at("type").s() == "text") systext += blk.at("text").s();
    if (!systext.empty()) msgs.push_back({"system", systext});
  }
  int merged = build_msgs(req.at("messages"), msgs, gh, gw);
  int max_new = req.has("max_tokens") ? (int)req.at("max_tokens").i64(g_max_tokens) : g_max_tokens;
  if (max_new <= 0 || max_new > 2048) max_new = g_max_tokens;
  bool stream = req.has("stream") && req.at("stream").boolean(false);
  bool enable_thinking = req.has("enable_thinking") && req.at("enable_thinking").boolean(false);

  std::string prompt = g_tok->apply_chat_template(msgs, true, enable_thinking);
  std::vector<int> ids = g_tok->encode(prompt, true);
  int prompt_tok = (int)ids.size();
  int eos = g_tok->eos();
  char idbuf[64]; snprintf(idbuf, 64, "msg_%ld", now_s());

  if (!stream) {
    std::vector<int> out = merged > 0 ? g_eng->generate_mm(ids, IMAGE_TOKEN_ID, gh, gw, max_new, eos)
                                      : g_eng->generate_spec(ids, max_new, eos, 1);   // MTP = faster
    std::string text = g_tok->decode(out, true);
    const char* stop_reason = (int)out.size() >= max_new ? "max_tokens" : "end_turn";
    std::string body = std::string("{\"id\":\"") + idbuf + "\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"" +
      g_model + "\",\"content\":[{\"type\":\"text\",\"text\":\"" + json_escape(text) + "\"}],\"stop_reason\":\"" +
      stop_reason + "\",\"stop_sequence\":null,\"usage\":{\"input_tokens\":" + std::to_string(prompt_tok) +
      ",\"output_tokens\":" + std::to_string((int)out.size()) + "}}";
    send_all(fd, http_json(body));
    return;
  }

  // streaming: Anthropic SSE event sequence (event: <name>\ndata: {...}\n\n)
  send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
               "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
  auto ev = [&](const char* name, const std::string& data) {
    send_all(fd, std::string("event: ") + name + "\ndata: " + data + "\n\n");
  };
  ev("message_start", std::string("{\"type\":\"message_start\",\"message\":{\"id\":\"") + idbuf +
     "\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"" + g_model +
     "\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":" + std::to_string(prompt_tok) +
     ",\"output_tokens\":0}}}");
  ev("content_block_start", "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}");
  ev("ping", "{\"type\":\"ping\"}");
  std::vector<int> acc;
  std::string sent;
  auto emit = [&](const std::string& delta) {
    ev("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"" +
       json_escape(delta) + "\"}}");
  };
  auto cb = [&](int tid) {
    acc.push_back(tid);
    std::string full = g_tok->decode(acc, true);
    size_t end = utf8_safe_end(full);
    if (end > sent.size()) { emit(full.substr(sent.size(), end - sent.size())); sent = full.substr(0, end); }
  };
  if (merged > 0) g_eng->generate_mm(ids, IMAGE_TOKEN_ID, gh, gw, max_new, eos, cb);
  else g_eng->generate_spec(ids, max_new, eos, 1, cb);   // MTP speculative decode (faster single-stream)
  std::string full = g_tok->decode(acc, true);
  if (full.size() > sent.size()) emit(full.substr(sent.size()));
  ev("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":0}");
  const char* sr = (int)acc.size() >= max_new ? "max_tokens" : "end_turn";
  ev("message_delta", std::string("{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"") + sr +
     "\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":" + std::to_string((int)acc.size()) + "}}");
  ev("message_stop", "{\"type\":\"message_stop\"}");
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
  if (method == "POST" && (path == "/v1/chat/completions" || path == "/v1/messages")) {
    try {
      Json r = json_parse(req.data() + body_start, clen);
      if (path == "/v1/messages") handle_messages(fd, r);   // Anthropic native
      else handle_chat(fd, r);                              // OpenAI
    } catch (std::exception& e) {
      send_all(fd, http_json(std::string("{\"error\":{\"type\":\"invalid_request_error\",\"message\":\"") +
                             json_escape(e.what()) + "\"}}", 400));
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
