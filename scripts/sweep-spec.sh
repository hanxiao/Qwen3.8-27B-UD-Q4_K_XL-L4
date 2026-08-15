#!/bin/bash
# Speculative-decoding parameter sweep. Reproduces the tuning table in the README.
#
# Scope: only DRAFT-SIDE and scheduling knobs are swept. Target-side KV quantization
# (--cache-type-k/v) is deliberately excluded because it changes model quality; this
# sweep is for finding speed at fixed quality.
#
# Run ON the instance (needs bash 4 + bc + a running /opt/models/model.gguf):
#   gcloud compute scp scripts/sweep-spec.sh <inst>:~/ --zone=<zone>
#   gcloud compute ssh <inst> --zone=<zone> --command 'sudo apt-get install -y -qq bc; bash ~/sweep-spec.sh'
#
# Each config restarts the server, warms up, then runs every workload twice and keeps
# the better pass (cold-container noise is ~1-2 tok/s and would otherwise dominate).
set -u
MODEL=/models/model.gguf
BASE_ARGS="--model $MODEL --alias sweep --host 0.0.0.0 --port 8080 --jinja --tools all \
--ctx-size ${CTX:-65536} --parallel 1 --flash-attn on -ngl 99 --no-mmap --threads 8 --no-warmup --metrics"

run_cfg() {
  local name="$1"; shift
  local extra="$*"
  sudo docker rm -f llama-server >/dev/null 2>&1
  sudo docker run -d --name llama-server --gpus all -p 8080:8080 -v /opt/models:/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda $BASE_ARGS $extra >/dev/null 2>&1
  for i in $(seq 1 60); do
    curl -fsS --max-time 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && break
    sleep 2
  done
  curl -fsS --max-time 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok \
    || { echo "$name|FAILED_TO_START"; return; }
  for w in 1 2; do
    curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0,"cache_prompt":false}' >/dev/null 2>&1
  done

  local best=0 bestout=""
  for pass in 1 2; do
    local tot=0 n=0 out=""
    for k in code prose chat math json; do
      case $k in
        code)  P="Write a complete Python implementation of a thread-safe LRU cache with get/put, type hints, docstrings, and 5 unit tests." ;;
        prose) P="Write three detailed paragraphs explaining how photosynthesis works in plants, from light absorption to glucose production." ;;
        chat)  P="I'm planning a 5-day trip to Tokyo in spring. Suggest a day-by-day itinerary with food recommendations and travel tips." ;;
        math)  P="Solve step by step: a train leaves city A at 60 mph, another leaves city B (300 miles away) at 40 mph toward A. When and where do they meet? Show all reasoning." ;;
        json)  P="Output a JSON array of 25 fictional books, each with title, author, year, genre, isbn, pages. Valid JSON only." ;;
      esac
      R=$(curl -s --max-time 180 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
          -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":256,\"temperature\":0,\"cache_prompt\":false}")
      V=$(echo "$R" | python3 -c "import sys,json
try:
 t=json.load(sys.stdin)['timings']; dn=t.get('draft_n',0)
 print(f\"{t['predicted_per_second']:.2f}:{(t.get('draft_n_accepted',0)/dn) if dn else 0:.3f}\")
except Exception: print('0:0')" 2>/dev/null)
      sp=${V%%:*}; ac=${V##*:}
      out="$out $k=$sp/$ac"
      tot=$(echo "$tot+$sp"|bc -l); n=$((n+1))
    done
    avg=$(echo "scale=2;$tot/$n"|bc -l)
    if (( $(echo "$avg > $best"|bc -l) )); then best=$avg; bestout="$out"; fi
  done
  echo "$name|BEST=$best|$bestout"
}

echo "=== draft depth ==="
for n in 2 3 4 5 6; do
  run_cfg "mtp_n$n            " "-ub 64 -b 512 --spec-type draft-mtp --spec-draft-n-max $n"
done

echo "=== ngram stacking (--spec-type takes a comma-separated list) ==="
run_cfg "mtp2+ngram-cache   " "-ub 64 -b 512 --spec-type draft-mtp,ngram-cache  --spec-draft-n-max 2"
run_cfg "mtp2+ngram-simple  " "-ub 64 -b 512 --spec-type draft-mtp,ngram-simple --spec-draft-n-max 2"
run_cfg "mtp2+ngram-map-k   " "-ub 64 -b 512 --spec-type draft-mtp,ngram-map-k  --spec-draft-n-max 2"
run_cfg "ngram-cache_only   " "-ub 64 -b 512 --spec-type ngram-cache            --spec-draft-n-max 2"

echo "=== p-min: skip drafting when the draft head is not confident ==="
for p in 0.2 0.3 0.35 0.4 0.45 0.5 0.6 0.8; do
  run_cfg "mtp2_pmin$p       " "-ub 64 -b 512 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min $p"
done

echo "=== deeper draft, now gated by p-min ==="
run_cfg "n3_pmin0.4         " "-ub 64 -b 512 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.4"
run_cfg "n4_pmin0.4         " "-ub 64 -b 512 --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.4"
run_cfg "n3_pmin0.5         " "-ub 64 -b 512 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.5"

echo "=== batch geometry ==="
run_cfg "ub128              " "-ub 128 -b 512  --spec-type draft-mtp --spec-draft-n-max 2"
run_cfg "ub256              " "-ub 256 -b 1024 --spec-type draft-mtp --spec-draft-n-max 2"
run_cfg "ub512              " "-ub 512 -b 2048 --spec-type draft-mtp --spec-draft-n-max 2"
