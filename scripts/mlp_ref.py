"""Reference for the MLP block y = down(silu(gate(x)) * up(x)) using the real
layer-0 NVFP4 weights, to verify the composed CUDA MLP (gemm+swiglu+gemm).
Saves x and y to test/mlp_x.f32 / test/mlp_yref.f32.

  docker run --rm -v /home/neo/models:/models -v $PWD:/work \
    vllm-spark:dev210-fast python3 /work/scripts/mlp_ref.py
"""
import json, struct, numpy as np, torch

MODEL = "/models/Qwen3.6-27B-NVFP4/model.safetensors"
L = "model.language_model.layers.0.mlp."
M = 2
LUT = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)

f = open(MODEL, "rb")
n = struct.unpack("<Q", f.read(8))[0]
H = json.loads(f.read(n))
base = 8 + n

def dequant(pfx):
    def rd(name):
        t = H[name]; b, e = t["data_offsets"]; f.seek(base + b); return t, f.read(e - b)
    pt, praw = rd(pfx + "weight_packed")
    st, sraw = rd(pfx + "weight_scale")
    gt, graw = rd(pfx + "weight_global_scale")
    OUT, IN2 = pt["shape"]; IN = IN2 * 2; G = 16
    packed = np.frombuffer(praw, np.uint8).reshape(OUT, IN2)
    scale = torch.tensor(np.frombuffer(sraw, np.uint8).reshape(OUT, IN // G).copy()) \
        .view(torch.float8_e4m3fn).float().numpy()
    gscale = float(np.frombuffer(graw, np.float32)[0])
    w = np.empty((OUT, IN), np.float32)
    w[:, 0::2] = np.where(packed & 8, -1., 1.) * LUT[packed & 7]
    w[:, 1::2] = np.where((packed >> 4) & 8, -1., 1.) * LUT[(packed >> 4) & 7]
    return (w * np.repeat(scale, G, axis=1) / gscale).astype(np.float32)

gate = dequant(L + "gate_proj.")   # [17408, 5120]
up = dequant(L + "up_proj.")       # [17408, 5120]
down = dequant(L + "down_proj.")   # [5120, 17408]
IN = gate.shape[1]
x = (np.sin(0.001 * (np.arange(M * IN).reshape(M, IN) + 1)) * 0.5).astype(np.float32)

g = x @ gate.T                     # [M, 17408]
u = x @ up.T
silu = g / (1.0 + np.exp(-g))
h = (silu * u).astype(np.float32)
y = (h @ down.T).astype(np.float32)  # [M, 5120]

x.tofile("/work/test/mlp_x.f32")
y.tofile("/work/test/mlp_yref.f32")
print("MLP ref: x[%d,%d] gate/up[%s] down[%s] -> y[%d,%d] |y|mean=%.4f"
      % (M, IN, gate.shape, down.shape, y.shape[0], y.shape[1], np.abs(y).mean()))
