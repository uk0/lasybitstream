"""Reference for the GDN gated-delta-net recurrence core, matching the exact
fla `fused_recurrent_gated_delta_rule` per-token update:
    q,k L2-normed; q*=scale
    S *= exp(g)                 # g = gating decay (given here as input)
    v -= S @ k                  # delta
    v *= beta                   # beta = sigmoid gate (given as input)
    S += outer(v, k)
    out = S @ q
per v-head, state S[head_v, head_k]. GQA: HV v-heads share HK k-heads (HV/HK each).
Saves synthetic inputs + output as raw f32 for the CUDA kernel test.

  docker run --rm -v $PWD:/work vllm-spark:dev210-fast python3 /work/scripts/gdn_ref.py
"""
import numpy as np, torch

torch.manual_seed(0)
T, HK, HV, DK, DV = 8, 16, 48, 128, 128   # tokens, k-heads, v-heads, head_k, head_v
SCALE = 1.0
OUT = "/work/test"

q = (torch.randn(T, HK, DK) * 0.5)
k = (torch.randn(T, HK, DK) * 0.5)
v = (torch.randn(T, HV, DV) * 0.5)
g = (-torch.rand(T, HV) * 0.1)            # decay (negative log-rate)
beta = torch.rand(T, HV)                  # gate in (0,1)

kh = torch.arange(HV) // (HV // HK)       # k-head index for each v-head
S = torch.zeros(HV, DV, DK)
out = torch.zeros(T, HV, DV)
for t in range(T):
    bq = q[t, kh]                          # [HV, DK]
    bk = k[t, kh]
    bq = bq / torch.sqrt((bq * bq).sum(-1, keepdim=True) + 1e-6)
    bk = bk / torch.sqrt((bk * bk).sum(-1, keepdim=True) + 1e-6)
    bq = bq * SCALE
    S = S * torch.exp(g[t])[:, None, None]                 # decay
    vpred = (S * bk[:, None, :]).sum(-1)                    # [HV, DV] = S @ k
    bv = (v[t] - vpred) * beta[t][:, None]
    S = S + bv[:, :, None] * bk[:, None, :]                 # outer(v, k)
    out[t] = (S * bq[:, None, :]).sum(-1)                   # S @ q

for name, arr in [("gdn_q", q), ("gdn_k", k), ("gdn_v", v), ("gdn_g", g),
                  ("gdn_beta", beta), ("gdn_out", out)]:
    arr.numpy().astype(np.float32).tofile(OUT + "/%s.f32" % name)
print("GDN recurrence ref: T=%d HK=%d HV=%d DK=%d DV=%d scale=%.1f |out|mean=%.4f -> test/gdn_*.f32"
      % (T, HK, HV, DK, DV, SCALE, out.abs().mean().item()))
