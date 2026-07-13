#!/usr/bin/env python3
"""End-to-end demo for the lasybitstream server: multi-turn chat on both the
OpenAI (/v1/chat/completions) and Anthropic (/v1/messages) APIs, plus multimodal
input with a base64 image and an image URL.

Start the server first:
    ./build/lbserve /path/to/Qwen3.6-27B-NVFP4 8080 128

Then:  python3 examples/demo.py [host:port]  (default 127.0.0.1:8080)
No third-party deps — uses urllib and a tiny pure-Python PNG encoder.
"""
import sys, json, base64, struct, zlib, urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8080"
BASE = f"http://{HOST}"


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def make_png(w=64, h=64):
    """A tiny RGB gradient PNG, base64-encoded — no PIL."""
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0
        for x in range(w):
            raw += bytes(((x * 4) % 256, (y * 4) % 256, ((x + y) * 2) % 256))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


def oai_text(msg): return post("/v1/chat/completions", msg)["choices"][0]["message"]["content"]
def ant_text(msg): return post("/v1/messages", msg)["content"][0]["text"]


print("=== 1) OpenAI multi-turn chat ===")
hist = [{"role": "user", "content": "My name is Ada. Remember it."}]
r1 = oai_text({"messages": hist, "max_tokens": 40, "enable_thinking": False})
print("  turn1:", r1.strip()[:120])
hist += [{"role": "assistant", "content": r1}, {"role": "user", "content": "What is my name?"}]
r2 = oai_text({"messages": hist, "max_tokens": 30, "enable_thinking": False})
print("  turn2:", r2.strip()[:120])

print("=== 2) Anthropic multi-turn (/v1/messages) ===")
hist = [{"role": "user", "content": "In one word, the capital of Japan?"}]
a1 = ant_text({"messages": hist, "max_tokens": 16})
print("  turn1:", a1.strip()[:120])
hist += [{"role": "assistant", "content": a1}, {"role": "user", "content": "And of France?"}]
a2 = ant_text({"messages": hist, "max_tokens": 16})
print("  turn2:", a2.strip()[:120])

print("=== 3) OpenAI multimodal — base64 image ===")
img_b64 = make_png()
r = oai_text({"max_tokens": 48, "enable_thinking": False, "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image briefly."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}}]}]})
print("  ", r.strip()[:200])

print("=== 4) Anthropic multimodal — base64 image ===")
r = ant_text({"max_tokens": 48, "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What does this image show?"},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}}]}]})
print("  ", r.strip()[:200])

print("=== 5) OpenAI multimodal — image URL ===")
try:
    r = oai_text({"max_tokens": 48, "enable_thinking": False, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png"}}]}]})
    print("  ", r.strip()[:200])
except Exception as e:
    print("  (URL fetch skipped:", e, ")")

print("\n== demo complete ==")
