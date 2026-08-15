#!/bin/bash
# Binary search for the largest --ctx-size that loads and serves on this GPU.
#
# A context is accepted only if the server reaches /health AND completes a real
# generation. Loading alone is not sufficient: llama.cpp allocates the KV cache at load
# time but some compute buffers are sized on first decode, so a context that loads can
# still fail under traffic.
#
# Run ON the instance:
#   gcloud compute scp scripts/max-context.sh <inst>:~/ --zone=<zone>
#   gcloud compute ssh <inst> --zone=<zone> --command 'bash ~/max-context.sh'
#
# Env: LO (known-good, default 65536), HI (known-bad upper bound, default 262144),
#      TOL (stop when HI-LO <= TOL, default 256), SPEC (extra server args)
set -u
LO="${LO:-65536}"
HI="${HI:-262144}"
TOL="${TOL:-256}"
SPEC="${SPEC:---spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.4}"

try_ctx() {
  local ctx="$1"
  sudo docker rm -f llama-server >/dev/null 2>&1
  sudo docker run -d --name llama-server --gpus all -p 8080:8080 -v /opt/models:/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda --model /models/model.gguf --alias ctxprobe \
    --host 0.0.0.0 --port 8080 --jinja --tools all --ctx-size "$ctx" --parallel 1 \
    --flash-attn on -ngl 99 -ub 64 -b 512 --no-mmap --threads 8 $SPEC \
    --no-warmup --metrics >/dev/null 2>&1

  # Wait for health, but bail early if the container has already died.
  local up=0
  for i in $(seq 1 90); do
    if ! sudo docker ps --format '{{.Names}}' | grep -q '^llama-server$'; then
      return 1
    fi
    if curl -fsS --max-time 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok; then
      up=1; break
    fi
    sleep 2
  done
  [ "$up" = 1 ] || return 1

  # A real generation, because compute buffers are sized on first decode.
  local out
  out=$(curl -s --max-time 120 http://127.0.0.1:8080/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Count from 1 to 20."}],"max_tokens":64,"temperature":0,"cache_prompt":false}' \
        2>/dev/null | python3 -c "
import sys,json
try:
  d=json.load(sys.stdin)
  c=d['choices'][0]['message']
  t=(c.get('content') or '')+(c.get('reasoning_content') or '')
  print('OK' if len(t.strip())>0 else 'EMPTY')
except Exception: print('ERR')" 2>/dev/null)
  [ "$out" = "OK" ]
}

vram() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' '
}

echo "Binary search for max --ctx-size"
echo "  known good: $LO    upper bound: $HI    tolerance: $TOL"
echo

# Confirm the lower bound really is good before trusting the search.
printf 'verify LO=%-7s ... ' "$LO"
if try_ctx "$LO"; then echo "OK  ($(vram))"; else echo "FAILED - LO is not good, aborting"; exit 1; fi

# Confirm HI actually fails; if it works we are done immediately.
printf 'probe   HI=%-7s ... ' "$HI"
if try_ctx "$HI"; then
  echo "OK  ($(vram))"
  echo
  echo "MAX >= $HI (model native limit reached, nothing to search)"
  exit 0
else
  echo "fail"
fi

BEST="$LO"
while [ $(( HI - LO )) -gt "$TOL" ]; do
  MID=$(( (LO + HI) / 2 ))
  # round down to a multiple of 256 for tidy numbers
  MID=$(( MID / 256 * 256 ))
  [ "$MID" -le "$LO" ] && break
  printf 'try     %-10s ... ' "$MID"
  if try_ctx "$MID"; then
    echo "OK  ($(vram))"
    BEST="$MID"; LO="$MID"
  else
    echo "fail"
    HI="$MID"
  fi
done

echo
echo "MAX WORKING CTX = $BEST"
echo "  (next size up, ~$HI, fails)"
echo
echo "Restoring server at the largest verified context ..."
try_ctx "$BEST" && echo "serving at ctx=$BEST, VRAM $(vram)"
