"""Fine-tune the nextn draft head on captured target hidden states, unrolled K steps.

Objective is acceptance: the head must emit what the TARGET would emit, so the label is the
target argmax at position i+1, and the soft target is the captured top-8 logits there. Restricting
the loss to those 8 ids removes the 248,320-wide lm_head from the backward pass, which is what
makes this tractable on one L4.

Teacher forcing only ever shows the head the TARGET's hidden state, but common/speculative.cpp
feeds the head its OWN pre-norm state forward (llama_get_embeddings_nextn_ith, line 818) for draft
positions 1..n. That mismatch is exposure bias, and it is where accepted length is lost: measured
per-position acceptance is 0.78, 0.55, 0.35, 0.25, 0.18. Unrolling K steps and supervising each one
trains the conditional acceptance the drafter is actually judged on. The token fed at step k is the
true token, which is correct: acceptance at position k is conditional on positions 0..k-1 already
matching, and in that case the drafted token equals the true one.
"""
import argparse, json, math, os, struct, sys, time, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--cap", default="/home/hanxiao/draft/cap-full.bin")
ap.add_argument("--hf", default="/home/hanxiao/draft/hf")
ap.add_argument("--train-pos", type=int, default=50_000)
ap.add_argument("--eval-pos", type=int, default=8_192)
ap.add_argument("--window", type=int, default=256)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--out", default="")
ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--anchor", type=float, default=0.0)
ap.add_argument("--unroll", type=int, default=4)
A = ap.parse_args()

NH, NKV, HD, ROT, THETA, EPS = 24, 4, 256, 64, 10_000_000.0, 1e-6
NE, VOC, TOPK = 5120, 248320, 8
REC = 4 + NE * 2 + 4 + TOPK * 8
dev = "cuda"
P = "mtp.layers.0."

def load_st(path, names):
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

WN = ["mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
      "mtp.norm.weight", P + "input_layernorm.weight", P + "post_attention_layernorm.weight",
      P + "self_attn.q_proj.weight", P + "self_attn.k_proj.weight", P + "self_attn.v_proj.weight",
      P + "self_attn.o_proj.weight", P + "self_attn.q_norm.weight", P + "self_attn.k_norm.weight",
      P + "mlp.gate_proj.weight", P + "mlp.up_proj.weight", P + "mlp.down_proj.weight"]
raw = load_st(os.path.join(A.hf, "mtp-shard.safetensors"), WN + ["lm_head.weight"])
emb = load_st(os.path.join(A.hf, "embed-shard.safetensors"),
              ["model.language_model.embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
LM  = raw["lm_head.weight"].to(dev, torch.bfloat16)
EMB = emb.to(dev, torch.bfloat16)
W = {k: raw[k].to(dev, torch.float32).clone().requires_grad_(True) for k in WN}
params = list(W.values())
W0 = {k: v.detach().clone() for k, v in W.items()}
print("  trainable %.1fM params over %d tensors" % (sum(p.numel() for p in params) / 1e6, len(params)))

def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w)

_d = ROT // 2
_inv = (1.0 / (THETA ** (torch.arange(0, _d, dtype=torch.float32, device=dev) * 2.0 / ROT)))
def rope(x, pos):
    ang = pos[:, None].float() * _inv[None, :]
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    r, keep = x[..., :ROT], x[..., ROT:]
    a, b = r[..., :_d], r[..., _d:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin, keep], -1)

def block(h, tok, pos, mask):
    B, T = tok.shape
    e_n = rms(EMB[tok].float(), W["mtp.pre_fc_norm_embedding.weight"])
    h_n = rms(h, W["mtp.pre_fc_norm_hidden.weight"])
    cur = torch.cat([e_n, h_n], -1) @ W["mtp.fc.weight"].T
    inpSA = cur
    x = rms(cur, W[P + "input_layernorm.weight"])
    qg = (x @ W[P + "self_attn.q_proj.weight"].T).view(B, T, NH, 2, HD)
    q, gate = qg[..., 0, :], qg[..., 1, :]
    k = (x @ W[P + "self_attn.k_proj.weight"].T).view(B, T, NKV, HD)
    v = (x @ W[P + "self_attn.v_proj.weight"].T).view(B, T, NKV, HD)
    q = rope(rms(q, W[P + "self_attn.q_norm.weight"]), pos)
    k = rope(rms(k, W[P + "self_attn.k_norm.weight"]), pos)
    k = k.repeat_interleave(NH // NKV, 2); v = v.repeat_interleave(NH // NKV, 2)
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       is_causal=True).transpose(1, 2)
    o = (o * torch.sigmoid(gate)).reshape(B, T, NH * HD) @ W[P + "self_attn.o_proj.weight"].T
    cur = o + inpSA; res = cur
    x = rms(cur, W[P + "post_attention_layernorm.weight"])
    g = x @ W[P + "mlp.gate_proj.weight"].T
    u = x @ W[P + "mlp.up_proj.weight"].T
    cur = (F.silu(g) * u) @ W[P + "mlp.down_proj.weight"].T + res
    return cur, rms(cur, W["mtp.norm.weight"])   # pre-norm state is what is fed forward

fsz = os.path.getsize(A.cap)
NPOS = (fsz - 20) // REC
fh = open(A.cap, "rb")
def read_window(start, T, K):
    """Returns h[T] plus tok/argmax/top8 for the next K+1 offsets, for a K-step unroll."""
    fh.seek(20 + start * REC)
    buf = fh.read(REC * (T + K + 1))
    n = T + K + 1
    tk = torch.empty(n, dtype=torch.long); am = torch.empty(n, dtype=torch.long)
    hs = torch.empty(n, NE, dtype=torch.bfloat16)
    ids = torch.empty(n, TOPK, dtype=torch.long); lg = torch.empty(n, TOPK)
    for t in range(n):
        o = t * REC
        tk[t] = struct.unpack_from("<i", buf, o)[0]
        hs[t] = torch.frombuffer(bytearray(buf[o + 4: o + 4 + NE * 2]), dtype=torch.bfloat16)
        am[t] = struct.unpack_from("<i", buf, o + 4 + NE * 2)[0]
        b = o + 8 + NE * 2
        ids[t] = torch.tensor(struct.unpack_from("<%di" % TOPK, buf, b))
        lg[t] = torch.tensor(struct.unpack_from("<%df" % TOPK, buf, b + TOPK * 4))
    return hs[:T], tk, am, ids, lg          # tok/am/ids/lg indexed from 0; step k uses [k+1 : k+1+T]

def batch(starts, T, K):
    H, TK, AM, ID, LG = zip(*[read_window(s, T, K) for s in starts])
    return (torch.stack(H).to(dev, torch.float32), torch.stack(TK).to(dev),
            torch.stack(AM).to(dev), torch.stack(ID).to(dev), torch.stack(LG).to(dev))

T, K = A.window, A.unroll
pos = torch.arange(T, device=dev)
g = torch.Generator().manual_seed(0)
EVAL_BASE = NPOS - A.eval_pos - T - K - 2
eval_starts = [EVAL_BASE + i * T for i in range(A.eval_pos // T)]
train_hi = EVAL_BASE - T - K - 2

@torch.no_grad()
def evaluate():
    """Per-step top-1, chaining the head's own pre-norm state exactly as the drafter does."""
    hit = [0] * K; tot = [0] * K
    for i in range(0, len(eval_starts), A.batch):
        st = eval_starts[i:i + A.batch]
        h, tk, am, ids, lg = batch(st, T, K)
        hin = h
        with torch.autocast("cuda", torch.bfloat16):
            for k in range(K):
                cur, fin = block(hin, tk[:, k + 1:k + 1 + T], pos, None)
                pr = (fin.reshape(-1, NE).bfloat16() @ LM.T).argmax(-1)
                a = am[:, k + 1:k + 1 + T].reshape(-1)
                m = a >= 0
                hit[k] += (pr[m] == a[m]).sum().item(); tot[k] += m.sum().item()
                hin = cur.float()
    return [hit[k] / max(tot[k], 1) for k in range(K)]

def fmt(v):
    return " ".join("%.3f" % x for x in v)

base = evaluate()
print("  baseline per-step top-1 on held-out: %s   (mean %.4f, %d positions)"
      % (fmt(base), sum(base) / len(base), A.eval_pos))
if A.train_pos == 0:
    sys.exit(0)

opt = torch.optim.AdamW(params, lr=A.lr, betas=(0.9, 0.95), weight_decay=0.0)
steps = A.train_pos // (A.batch * T)
print("  training %d steps, batch %d x %d = %d positions/step, lr %.1e"
      % (steps, A.batch, T, A.batch * T, A.lr))
t0 = time.time(); best = sum(base) / len(base)
for s in range(steps):
    st = [int(torch.randint(0, train_hi, (1,), generator=g).item()) for _ in range(A.batch)]
    h, tk, am, ids, lg = batch(st, T, K)
    for gp in opt.param_groups:
        gp["lr"] = A.lr * min(1.0, (s + 1) / A.warmup) * (0.5 * (1 + math.cos(math.pi * s / max(steps, 1))) * 0.9 + 0.1)
    loss = 0.0
    hin = h
    with torch.autocast("cuda", torch.bfloat16):
        for k in range(K):
            cur, fin = block(hin, tk[:, k + 1:k + 1 + T], pos, None)
            idk = ids[:, k + 1:k + 1 + T].reshape(-1)
            sel = LM[idk].view(-1, TOPK, NE)
            mine = torch.einsum("bd,bkd->bk", fin.reshape(-1, NE).bfloat16(), sel).float()
            teach = lg[:, k + 1:k + 1 + T].reshape(-1, TOPK).float().softmax(-1)
            loss = loss + -(teach * mine.log_softmax(-1)).sum(-1).mean()
            hin = cur.float()
    loss = loss / K
    if A.anchor:
        loss = loss + A.anchor * sum(((W[k] - W0[k]) ** 2).mean() for k in W)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if (s + 1) % A.eval_every == 0 or s == steps - 1:
        a = evaluate()
        ma = sum(a) / len(a)
        best = max(best, ma)
        print("    step %4d/%d  loss %.4f  per-step %s  mean %.4f (%+.4f)  %.0fs"
              % (s + 1, steps, loss.item(), fmt(a), ma, ma - sum(base) / len(base), time.time() - t0), flush=True)
print("  baseline mean %.4f  ->  best %.4f  (%+.4f)" % (sum(base) / len(base), best, best - sum(base) / len(base)))
if A.out:
    torch.save({k: v.detach().cpu() for k, v in W.items()}, A.out)
    print("  saved %s" % A.out)
