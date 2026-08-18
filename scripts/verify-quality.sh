#!/usr/bin/env bash
# Quality verification for the tuned configuration. Two independent tests, because the
# first one alone is misleading.
#
#   1. Greedy determinism  -- byte-compare output against --spec-type none.
#   2. Verifiable accuracy -- 40 arithmetic problems with known ground truth.
#
# Test 1 is EXPECTED TO FAIL under speculative decoding, and that is not a regression.
# Accepting a drafted token changes the batch shape of the forward pass, which changes
# float reduction order, which flips tokens that were near-ties. Test 2 decides whether
# quality actually moved.
#
# Run ON the instance:  bash ~/verify-quality.sh
# Env: BIN (dir holding bin/llama-server), MODEL, DRAFT, CTX
set -u
BIN="${BIN:-/opt/llama.cpp/build}"
MODEL="${MODEL:-/opt/models/model.gguf}"
DRAFT="${DRAFT:-/opt/models/mtp-o2k.gguf}"
CTX="${CTX:-8192}"
CHAIN="GGML_MMVQ_MAX=2,GGML_MMVQ_MAX_Q6K=8,GGML_MMVQ_MAX_IQ4XS=8,LLAMA_SPEC_CHAIN=1,LLAMA_SPEC_CHAIN_SUB=98304,LLAMA_SCHED_POOL=8"

cat > /tmp/_det.py <<'PY'
import json,urllib.request,sys
IP="127.0.0.1:8080"
PROMPTS=[
 ("code","Write a Python function to merge two sorted lists. Include a docstring."),
 ("math","Compute 47*83 and then explain the multiplication step by step."),
 ("json","Output a JSON object with keys name, age, city for a fictional person. JSON only."),
 ("prose","Explain in one paragraph why the sky appears blue."),
 ("logic","If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Explain."),
]
def call(p):
    body={"messages":[{"role":"user","content":p}],"max_tokens":400,"temperature":0,
          "seed":42,"cache_prompt":False}
    r=urllib.request.urlopen(urllib.request.Request(f"http://{IP}/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600)
    m=json.load(r)["choices"][0]["message"]
    return (m.get("reasoning_content") or "")+"|||"+(m.get("content") or "")
json.dump({k:call(p) for k,p in PROMPTS}, open(sys.argv[1],"w"))
PY

cat > /tmp/_acc.py <<'PY'
import json,urllib.request,random
IP="127.0.0.1:8080"
random.seed(7)
Q=[]
for _ in range(40):
    a=random.randint(11,99); b=random.randint(11,99)
    Q.append((f"Compute {a}*{b}. Reply with ONLY the integer, no words.", a*b))
ok=0; bad=[]
for q,gt in Q:
    body={"messages":[{"role":"user","content":q}],"max_tokens":900,"temperature":0,
          "seed":42,"cache_prompt":False,"chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.urlopen(urllib.request.Request(f"http://{IP}/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600)
    c=json.load(r)["choices"][0]["message"].get("content") or ""
    d="".join(ch for ch in c if ch.isdigit() or ch=="-")
    good = d.strip()==str(gt)
    ok+=good
    if not good: bad.append((q,gt,c[:60]))
print(f"  ACCURACY {ok}/{len(Q)} = {100*ok/len(Q):.1f}%")
for b in bad[:5]: print("    MISS:",b)
PY

start() {
  local envs="$1"; shift
  pkill -f "bin/llama-server" 2>/dev/null
  for i in $(seq 1 60); do pgrep -f "bin/llama-server" >/dev/null || break; sleep 1; done
  pkill -9 -f "bin/llama-server" 2>/dev/null
  for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":8080 " || break; sleep 1; done
  sleep 1
  ( cd "$BIN"
    export LD_LIBRARY_PATH="$BIN/bin:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
    if [ "$envs" != "-" ]; then IFS=',' read -ra E <<< "$envs"; for e in "${E[@]}"; do export "$e"; done; fi
    setsid nohup ./bin/llama-server --model "$MODEL" --alias v --host 0.0.0.0 --port 8080 \
      --jinja --tools all --ctx-size "$CTX" --parallel 1 --flash-attn on -ngl 99 \
      -ub 512 -b 512 --no-mmap --threads 8 --no-warmup --metrics "$@" \
      > /tmp/vq-server.log 2>&1 < /dev/null & )
  for i in $(seq 1 240); do
    curl -fsS --max-time 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && return 0
    sleep 1
  done
  echo "  server failed to start"; return 1
}

echo "== reference: no speculation, stock kernel selection =="
start - --spec-type none
python3 /tmp/_det.py /tmp/out_nospec.json
echo "-- control: same config twice, is the server deterministic at all?"
python3 /tmp/_det.py /tmp/out_nospec2.json
python3 - <<'PY'
import json
a=json.load(open("/tmp/out_nospec.json")); b=json.load(open("/tmp/out_nospec2.json"))
print("   run-to-run deterministic:", all(a[k]==b[k] for k in a))
PY
echo "-- accuracy (reference)"
python3 /tmp/_acc.py

echo
echo "== kernel routing only, no speculation =="
start "$CHAIN" --spec-type none
python3 /tmp/_det.py /tmp/out_mmq.json
echo "-- accuracy"
python3 /tmp/_acc.py

echo
echo "== shipped config: kernel routing + chain drafting + MTP sidecar n=5 =="
start "$CHAIN" --spec-type draft-mtp --spec-draft-model "$DRAFT" \
  --spec-draft-ngl 99 --spec-draft-n-max 5 --spec-draft-n-min 5 --spec-draft-p-min 0.0 -bs
python3 /tmp/_det.py /tmp/out_opt.json
echo "-- accuracy"
python3 /tmp/_acc.py

echo
echo "== byte-comparison =="
python3 - <<'PY'
import json
ref=json.load(open("/tmp/out_nospec.json"))
mmq=json.load(open("/tmp/out_mmq.json"))
opt=json.load(open("/tmp/out_opt.json"))
print("  task    mmq_vs_nospec   tuned_vs_nospec")
for k in ref:
    print("  %-6s  %-14s  %-14s" % (k, mmq[k]==ref[k], opt[k]==ref[k]))
print()
print("  Byte divergence under speculation is expected; judge on accuracy, not bytes.")
print("  mmq_vs_nospec is the interesting column: it isolates the kernel change from")
print("  speculation, and should be True if the MMQ path is numerically equivalent.")
PY
