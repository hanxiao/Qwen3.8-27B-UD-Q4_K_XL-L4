"""Reference forward pass for the Qwen3.8-27B nextn (MTP) draft head, in torch.

Reproduces what llama.cpp runs in src/models/qwen35.cpp build_mtp, so the head can be trained or
inspected outside llama.cpp. Scores 68.8% top-1 on tok_{i+2} given (h_i, tok_{i+1}) against
captured hidden states; llama.cpp reports 0.774 acceptance at draft position 0 in production.

Two conventions matter and neither is visible from the HF checkpoint alone:

  1. Every RMSNorm weight here is stored as an OFFSET. The effective scale is `1 + w`, which is
     what convert_hf_to_gguf.py bakes into the GGUF. Using raw `w` negates the block input
     (mtp.pre_fc_norm_embedding is 100% negative, mean -0.4606) and top-1 collapses to 0.0% with
     the true token anti-correlated at median rank 247,841 of 248,320.
  2. q_proj packs Q and the attention gate INTERLEAVED per head at head_dim*2 stride.

transformers never loads these tensors (_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]), so
there is no upstream implementation to check against.

Usage: mtp_head_reference.py <capture.bin> <mtp-shard.safetensors> <embed-shard.safetensors>
"""
import struct, sys, torch, torch.nn.functional as F

def load_st(path, names):
    """Minimal safetensors reader; avoids a dependency for one function."""
    import json
    DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n)); base = 8 + n
        for k in names:
            m = hdr[k]; s, e = m["data_offsets"]
            f.seek(base + s); raw = bytearray(f.read(e - s))
            out[k] = torch.frombuffer(raw, dtype=DT[m["dtype"]]).view(*m["shape"])
    return out

NH, NKV, HD, ROT, THETA, EPS = 24, 4, 256, 64, 10_000_000.0, 1e-6
W = 512                      # causal window
p = "mtp.layers.0."

def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w.float())

def rope(x, pos, half=True):
    """x: [T, H, HD]. Rotate first ROT dims, pass the rest through."""
    d = ROT // 2
    inv = 1.0 / (THETA ** (torch.arange(0, d, dtype=torch.float32) * 2.0 / ROT))
    ang = pos[:, None].float() * inv[None, :]           # [T, d]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    r, keep = x[..., :ROT], x[..., ROT:]
    if half:                                            # HF rotate_half: (i, i+d)
        a, b = r[..., :d], r[..., d:]
        out = torch.cat([a * cos - b * sin, b * cos + a * sin], -1)
    else:                                               # interleaved: (2i, 2i+1)
        a, b = r[..., 0::2], r[..., 1::2]
        ra, rb = a * cos - b * sin, b * cos + a * sin
        out = torch.stack([ra, rb], -1).flatten(-2)
    return torch.cat([out, keep], -1)

cap, MTP_SHARD, EMB_SHARD = sys.argv[1], sys.argv[2], sys.argv[3]
f = open(cap, "rb"); f.read(8)
n_embd, topk, npos = struct.unpack("<3i", f.read(12))
rec = 4 + n_embd * 2 + 4 + topk * 8
N = W + 2
toks, hs = [], []
for _ in range(N):
    b = f.read(rec)
    toks.append(struct.unpack_from("<i", b, 0)[0])
    hs.append(torch.frombuffer(bytearray(b[4:4 + n_embd * 2]), dtype=torch.bfloat16).float())

w = load_st(MTP_SHARD, [
    "mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight", p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
    p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight",
    p + "self_attn.o_proj.weight", p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight",
    p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight"])
emb = load_st(EMB_SHARD,
              ["model.language_model.embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
lm = load_st(MTP_SHARD, ["lm_head.weight"])["lm_head.weight"]

H    = torch.stack(hs[:W])
nxt  = torch.tensor(toks[1:W + 1])       # tok_{i+1}, the token fed alongside h_i
tgt  = torch.tensor(toks[2:W + 2])       # tok_{i+2}, what the head must predict
pos  = torch.arange(W)
mask = torch.full((W, W), float("-inf")).triu(1)

def run(half=True, gate_blocked=False):
    e_n = rms(emb[nxt].float(), w["mtp.pre_fc_norm_embedding.weight"])
    h_n = rms(H, w["mtp.pre_fc_norm_hidden.weight"])
    cur = torch.cat([e_n, h_n], -1) @ w["mtp.fc.weight"].float().T
    inpSA = cur
    x = rms(cur, w[p + "input_layernorm.weight"])

    qg = x @ w[p + "self_attn.q_proj.weight"].float().T
    if gate_blocked:
        qg = qg.view(W, 2, NH, HD); q, gate = qg[:, 0], qg[:, 1]
    else:
        qg = qg.view(W, NH, 2, HD); q, gate = qg[:, :, 0], qg[:, :, 1]
    k = (x @ w[p + "self_attn.k_proj.weight"].float().T).view(W, NKV, HD)
    v = (x @ w[p + "self_attn.v_proj.weight"].float().T).view(W, NKV, HD)

    q = rope(rms(q, w[p + "self_attn.q_norm.weight"]), pos, half)
    k = rope(rms(k, w[p + "self_attn.k_norm.weight"]), pos, half)

    k = k.repeat_interleave(NH // NKV, 1); v = v.repeat_interleave(NH // NKV, 1)
    att = (torch.einsum("ihd,jhd->hij", q, k) / (HD ** 0.5) + mask).softmax(-1)
    o = torch.einsum("hij,jhd->ihd", att, v)
    o = (o * torch.sigmoid(gate)).reshape(W, NH * HD) @ w[p + "self_attn.o_proj.weight"].float().T

    cur = o + inpSA; res = cur
    x = rms(cur, w[p + "post_attention_layernorm.weight"])
    g = x @ w[p + "mlp.gate_proj.weight"].float().T
    u = x @ w[p + "mlp.up_proj.weight"].float().T
    cur = (F.silu(g) * u) @ w[p + "mlp.down_proj.weight"].float().T + res
    fin = rms(cur, w["mtp.norm.weight"])

    logits = fin[32:] @ lm.float().T          # skip the first few, thin causal context
    pred = logits.argmax(-1)
    t = tgt[32:]
    top1 = (pred == t).float().mean().item()
    rank = (logits > logits.gather(1, t[:, None])).sum(1).float().median().item()
    return top1, rank

for half in (True, False):
    for gb in (False, True):
        a, r = run(half, gb)
        print("  rope=%-11s gate=%-11s  top1(tok_i+2)=%6.1f%%   median_rank=%8.0f"
              % ("rotate_half" if half else "interleaved", "blocked" if gb else "interleaved", a * 100, r))
