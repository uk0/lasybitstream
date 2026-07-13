"""Canonical NVFP4 dequant reference for the CUDA kernel test.

Dequants Qwen3.6-27B layer-0 mlp.gate_proj (first 256 rows) to raw f32 at
test/ref_gate0.f32. Convention verified against `compressed_tensors`:
  * low nibble = even element (lo_even)
  * w = e2m1(nibble) * fp8_e4m3(weight_scale) / weight_global_scale

Run inside a container that has torch (for fp8 decode), mounting the model and
the repo:
  docker run --rm -v /home/neo/models:/models -v $PWD:/work \
    vllm-spark:dev210-fast python3 /work/scripts/nvfp4_ref.py
"""
import json, struct, numpy as np, torch

MODEL = "/models/Qwen3.6-27B-NVFP4/model.safetensors"
PFX = "model.language_model.layers.0.mlp.gate_proj."
ROWS = 256

f = open(MODEL, "rb")
n = struct.unpack("<Q", f.read(8))[0]
H = json.loads(f.read(n))
base = 8 + n

def rd(name):
    t = H[name]; b, e = t["data_offsets"]; f.seek(base + b); return t, f.read(e - b)

pt, praw = rd(PFX + "weight_packed")
st, sraw = rd(PFX + "weight_scale")
gt, graw = rd(PFX + "weight_global_scale")
OUT, IN2 = pt["shape"]; IN = IN2 * 2; G = 16
packed = np.frombuffer(praw, np.uint8).reshape(OUT, IN2)[:ROWS]
scale = torch.tensor(np.frombuffer(sraw, np.uint8).reshape(OUT, IN // G)[:ROWS].copy()) \
    .view(torch.float8_e4m3fn).float().numpy()
gscale = float(np.frombuffer(graw, np.float32)[0])

LUT = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
dec = lambda c: np.where(c & 8, -1.0, 1.0) * LUT[c & 7]
w = np.empty((ROWS, IN), np.float32)
w[:, 0::2] = dec(packed & 0x0F)          # low nibble -> even
w[:, 1::2] = dec((packed >> 4) & 0x0F)   # high nibble -> odd
deq = (w * np.repeat(scale, G, axis=1) / gscale).astype(np.float32)

deq.tofile("/work/test/ref_gate0.f32")
print("OUT=%d IN=%d group=%d gscale=%.1f ROWS=%d |w|mean=%.5f -> test/ref_gate0.f32 (%d floats)"
      % (OUT, IN, G, gscale, ROWS, np.abs(deq).mean(), deq.size))
