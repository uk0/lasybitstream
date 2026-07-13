"""Golden reference forward for Qwen3.6-27B-NVFP4 (dense, Qwen3-Next arch).

Self-contained W4A16 torch forward built from the pieces the CUDA engine
already verifies bit-for-bit: NVFP4 dequant, Gemma RMSNorm, GDN gated-delta-net
recurrence, SwiGLU. It dumps per-layer hidden states + final logits so the native
lasybitstream forward can be checked layer-by-layer.

  docker run --rm --gpus all \
    -v /home/neo/models/Qwen3.6-27B-NVFP4:/model \
    -v $PWD:/work vllm-spark:dev210-fast \
    python3 /work/scripts/ref_forward.py

Writes /work/test/ref_*.f32 (+ ref_tokens.i32, ref_meta.json).
"""
import json, os, struct
import numpy as np, torch, torch.nn.functional as F
from safetensors import safe_open

MODEL = os.environ.get("MODEL_DIR", "/model")
OUT = "/work/test"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
torch.set_grad_enabled(False)
# True FP32 accumulation (NGC containers default TF32 on) so the reference matches
# the native engine's f32 kernels, not a lower-precision tensor-core matmul.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

cfg = json.load(open(f"{MODEL}/config.json"))
tc = cfg.get("text_config", cfg)
H = tc["hidden_size"]                 # 5120
NL = tc["num_hidden_layers"]          # 64
EPS = tc["rms_norm_eps"]              # 1e-6
FA_INT = tc["full_attention_interval"]  # 4
NH = tc["num_attention_heads"]        # 24
NKV = tc["num_key_value_heads"]       # 4
HD = tc["head_dim"]                   # 256
L_KH = tc["linear_num_key_heads"]     # 16
L_VH = tc["linear_num_value_heads"]   # 48
L_KD = tc["linear_key_head_dim"]      # 128
L_VD = tc["linear_value_head_dim"]    # 128
CONV = tc["linear_conv_kernel_dim"]   # 4
rp = tc["rope_parameters"]
THETA = rp["rope_theta"]              # 1e7
PART = rp["partial_rotary_factor"]    # 0.25
ROT = int(HD * PART)                  # 64
VOCAB = tc["vocab_size"]

# ---- NVFP4 dequant (matches cuda/nvfp4.cu, verified bit-exact) ----
E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=DEV)

def deq(w_packed, w_scale, w_gscale):
    out, inh = w_packed.shape
    inn = inh * 2
    p = w_packed.to(torch.int32)
    lo = p & 0xF; hi = (p >> 4) & 0xF
    w = torch.empty(out, inn, device=DEV, dtype=torch.float32)
    w[:, 0::2] = torch.where((lo & 8) > 0, -1., 1.) * E2M1[lo & 7]
    w[:, 1::2] = torch.where((hi & 8) > 0, -1., 1.) * E2M1[hi & 7]
    sc = w_scale.float().repeat_interleave(16, dim=1)       # [out, inn]
    return w * sc / w_gscale.float()

# ---- weight store ----
f = safe_open(f"{MODEL}/model.safetensors", framework="pt", device=DEV)
KEYS = set(f.keys())

def g(name):
    return f.get_tensor(name).to(DEV)

def linear_nvfp4(x, pfx):
    w = deq(g(pfx + ".weight_packed"), g(pfx + ".weight_scale"), g(pfx + ".weight_global_scale"))
    return x @ w.t()

def linear_bf16(x, pfx):
    return x @ g(pfx + ".weight").float().t()

def rmsnorm(x, w):                       # Gemma: *(1+w)
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
    return v * (1.0 + w.float())

def rope(q, k, pos):                     # partial NeoX, rotary_dim=ROT, theta=THETA
    # q [T,NH,HD], k [T,NKV,HD]
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, device=DEV).float() / ROT))
    ang = pos.float()[:, None] * inv[None, :]      # [T, ROT/2]
    cos = torch.cos(ang); sin = torch.sin(ang)     # [T, ROT/2]
    def apply(t):
        rot, keep = t[..., :ROT], t[..., ROT:]
        a, b = rot[..., :ROT // 2], rot[..., ROT // 2:]
        c = cos[:, None, :]; s = sin[:, None, :]
        ro = torch.cat([a * c - b * s, b * c + a * s], dim=-1)
        return torch.cat([ro, keep], dim=-1)
    return apply(q), apply(k)

def gdn(x_in, L, dump):
    pfx = f"model.language_model.layers.{L}.linear_attn"
    T = x_in.shape[0]
    qkv = linear_bf16(x_in, pfx + ".in_proj_qkv")     # [T,10240]
    z = linear_bf16(x_in, pfx + ".in_proj_z")         # [T,6144]
    a = linear_bf16(x_in, pfx + ".in_proj_a")         # [T,48]
    b = linear_bf16(x_in, pfx + ".in_proj_b")         # [T,48]
    # depthwise causal conv1d (kernel CONV) + SiLU on the 10240 mixed channels
    cw = g(pfx + ".conv1d.weight").float().squeeze(1)  # [10240, CONV]
    xT = qkv.t().unsqueeze(0)                          # [1,10240,T]
    xp = F.pad(xT, (CONV - 1, 0))
    conv = F.conv1d(xp, cw.unsqueeze(1), groups=qkv.shape[1])[0].t()  # [T,10240]
    qkv = F.silu(conv)
    q = qkv[:, :2048].view(T, L_KH, L_KD)
    k = qkv[:, 2048:4096].view(T, L_KH, L_KD)
    v = qkv[:, 4096:].view(T, L_VH, L_VD)
    # gating
    A_log = g(pfx + ".A_log").float(); dt = g(pfx + ".dt_bias").float()
    gg = -torch.exp(A_log) * F.softplus(a + dt)        # [T,48]
    beta = torch.sigmoid(b)                            # [T,48]
    # recurrence (fla fused_recurrent_gated_delta_rule), scale = L_KD**-0.5
    scale = L_KD ** -0.5
    qn = F.normalize(q, dim=-1) * scale
    kn = F.normalize(k, dim=-1)
    S = torch.zeros(L_VH, L_VD, L_KD, device=DEV)
    out = torch.empty(T, L_VH, L_VD, device=DEV)
    for t in range(T):
        gt = torch.exp(gg[t])                          # [48]
        S = S * gt[:, None, None]
        kh = kn[t].repeat_interleave(L_VH // L_KH, dim=0)   # [48,128]
        qh = qn[t].repeat_interleave(L_VH // L_KH, dim=0)
        vpred = (S * kh[:, None, :]).sum(-1)           # [48,128]
        vt = (v[t] - vpred) * beta[t][:, None]
        S = S + vt[:, :, None] * kh[:, None, :]
        out[t] = (S * qh[:, None, :]).sum(-1)
    # gated RMSNorm (plain weight, per head_v=128) with z gate
    nw = g(pfx + ".norm.weight").float()
    o = out.reshape(T * L_VH, L_VD)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * nw
    o = o * F.silu(z.reshape(T * L_VH, L_VD))
    o = o.reshape(T, L_VH * L_VD)                      # [T,6144]
    return linear_nvfp4(o, pfx + ".out_proj")

def attn(x_in, L, pos):
    pfx = f"model.language_model.layers.{L}.self_attn"
    T = x_in.shape[0]
    qg = linear_nvfp4(x_in, pfx + ".q_proj").view(T, NH, 2 * HD)
    q, gate = qg[..., :HD], qg[..., HD:]               # per-head split
    q = q.reshape(T, NH, HD); gate = gate.reshape(T, NH, HD)
    k = linear_nvfp4(x_in, pfx + ".k_proj").view(T, NKV, HD)
    v = linear_nvfp4(x_in, pfx + ".v_proj").view(T, NKV, HD)
    q = rmsnorm(q, g(pfx + ".q_norm.weight"))
    k = rmsnorm(k, g(pfx + ".k_norm.weight"))
    q, k = rope(q, k, pos)
    # GQA causal attention
    rep = NH // NKV
    kk = k.repeat_interleave(rep, dim=1)               # [T,NH,HD]
    vv = v.repeat_interleave(rep, dim=1)
    q = q.transpose(0, 1); kk = kk.transpose(0, 1); vv = vv.transpose(0, 1)  # [NH,T,HD]
    scores = (q @ kk.transpose(-1, -2)) * (HD ** -0.5)  # [NH,T,T]
    mask = torch.triu(torch.full((T, T), float("-inf"), device=DEV), 1)
    scores = scores + mask
    o = torch.softmax(scores, -1) @ vv                 # [NH,T,HD]
    o = o.transpose(0, 1)                              # [T,NH,HD]
    o = o * torch.sigmoid(gate)
    o = o.reshape(T, NH * HD)                          # [T,6144]
    return linear_nvfp4(o, pfx + ".o_proj")

def mlp(x_in, L):
    pfx = f"model.language_model.layers.{L}.mlp"
    gate = linear_nvfp4(x_in, pfx + ".gate_proj")
    up = linear_nvfp4(x_in, pfx + ".up_proj")
    return linear_nvfp4(F.silu(gate) * up, pfx + ".down_proj")

# ---- tokens ----
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
except Exception as e:
    print("tokenizer failed, using fixed ids:", e)
    ids = [785, 6722, 315, 9625, 374]
T = len(ids)
pos = torch.arange(T, device=DEV)
print("tokens:", ids, "T=", T)

# ---- forward ----
emb = g("model.language_model.embed_tokens.weight")
h = emb[torch.tensor(ids, device=DEV)].float()         # [T,H]
np.array(ids, dtype=np.int32).tofile(f"{OUT}/ref_tokens.i32")
h.cpu().numpy().astype(np.float32).tofile(f"{OUT}/ref_embed.f32")

for L in range(NL):
    ln = g(f"model.language_model.layers.{L}.input_layernorm.weight")
    xn = rmsnorm(h, ln)
    if (L % FA_INT) == (FA_INT - 1):
        mix = attn(xn, L, pos)
    else:
        mix = gdn(xn, L, L < 3)
    h = h + mix
    pln = g(f"model.language_model.layers.{L}.post_attention_layernorm.weight")
    h = h + mlp(rmsnorm(h, pln), L)
    if L < 4 or L == NL - 1:
        h.cpu().numpy().astype(np.float32).tofile(f"{OUT}/ref_h{L}.f32")

h = rmsnorm(h, g("model.language_model.norm.weight"))
h.cpu().numpy().astype(np.float32).tofile(f"{OUT}/ref_final.f32")
logits = h[-1:] @ g("lm_head.weight").float().t()      # [1,VOCAB]
logits.cpu().numpy().astype(np.float32).tofile(f"{OUT}/ref_logits.f32")
top = torch.topk(logits[0], 20)
print("argmax:", int(top.indices[0]), "top20 ids:", top.indices.tolist())
print("top20 val:", [round(float(x), 3) for x in top.values.tolist()])
try:
    print("decoded next:", repr(tok.decode([int(top.indices[0])])))
except Exception:
    pass
json.dump({"ids": ids, "T": T, "argmax": int(top.indices[0]),
           "top20": top.indices.tolist(), "H": H, "NL": NL, "vocab": VOCAB},
          open(f"{OUT}/ref_meta.json", "w"))
print("wrote", OUT)
