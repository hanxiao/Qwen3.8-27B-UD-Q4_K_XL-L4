#!/bin/bash
# Quality verification for a speed change. Two independent tests, because the first one
# alone is misleading.
#
#   1. Greedy determinism  -- byte-compare output against --spec-type none.
#   2. Verifiable accuracy -- 40 arithmetic problems with known ground truth.
#
# Test 1 is EXPECTED TO FAIL under speculative decoding, and that is not a regression.
# Accepting a drafted token changes the batch shape of the forward pass, which changes
# float reduction order, which flips tokens that were near-ties. Test 2 is the one that
# decides whether quality actually moved.
#
# Run ON the instance:
#   gcloud compute scp scripts/verify-quality.sh <inst>:~/ --zone=<zone>
#   gcloud compute ssh <inst> --zone=<zone> --command 'bash ~/verify-quality.sh'
set -u

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
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=300)
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
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=300)
    c=json.load(r)["choices"][0]["message"].get("content") or ""
    d="".join(ch for ch in c if ch.isdigit() or ch=="-")
    good = d.strip()==str(gt)
    ok+=good
    if not good: bad.append((q,gt,c[:60]))
print(f"  ACCURACY {ok}/{len(Q)} = {100*ok/len(Q):.1f}%")
for b in bad[:5]: print("    MISS:",b)
PY

start() {
  sudo docker rm -f llama-server >/dev/null 2>&1
  sudo docker run -d --name llama-server --gpus all -p 8080:8080 -v /opt/models:/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda --model /models/model.gguf --alias v \
    --host 0.0.0.0 --port 8080 --jinja --tools all --ctx-size ${CTX:-65536} --parallel 1 \
    --flash-attn on -ngl 99 -ub 64 -b 512 --no-mmap --threads 8 --no-warmup --metrics \
    "$@" >/dev/null 2>&1
  for i in $(seq 1 60); do
    curl -fsS --max-time 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && return
    sleep 2
  done
}

echo "== capturing: no speculation (reference) =="
start --spec-type none
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
echo "== capturing: plain MTP n=2 =="
start --spec-type draft-mtp --spec-draft-n-max 2
python3 /tmp/_det.py /tmp/out_mtp.json
echo "-- accuracy"
python3 /tmp/_acc.py

echo
echo "== capturing: MTP n=2 + p-min 0.4 (shipped config) =="
start --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.4
python3 /tmp/_det.py /tmp/out_opt.json
echo "-- accuracy"
python3 /tmp/_acc.py

echo
echo "== byte-comparison =="
python3 - <<'PY'
import json
ref=json.load(open("/tmp/out_nospec.json"))
mtp=json.load(open("/tmp/out_mtp.json"))
opt=json.load(open("/tmp/out_opt.json"))
print("  task    mtp_vs_nospec   opt_vs_mtp")
for k in ref:
    print("  %-6s  %-14s  %-12s" % (k, mtp[k]==ref[k], opt[k]==mtp[k]))
print()
print("  Byte divergence is expected under speculation; judge on accuracy, not bytes.")
PY
