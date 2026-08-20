#!/bin/bash
# Instance startup script. Reads its configuration from instance metadata so that this
# file needs no shell interpolation from the provisioning script.
set -e
exec >>/var/log/qwen-startup.log 2>&1
echo "[startup] boot $(date -u +%H:%M:%S)"

# -f matters twice over. Without it curl prints the metadata server's 404 HTML page for an
# unset key, and a non-empty value defeats every ${VAR:-default} below. With it curl exits 22
# on that 404, which under `set -e` aborts the script at the first key the caller did not set,
# so the failure has to be swallowed here rather than propagated.
md() { curl -sf -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true; }

# The context is planned by llama.cpp, not set here: --fit sizes it to the device with
# --fit-target MiB left free, which is what keeps the compute graph buffers allocated at
# decode time from exhausting the card.
FIT_TARGET="$(md qwen-fit-target)"; FIT_TARGET="${FIT_TARGET:-768}"
KV="$(md qwen-kv)";          KV="${KV:-q4_0}"
UB="$(md qwen-ub)";          UB="${UB:-256}"
NMAX="$(md qwen-nmax)";      NMAX="${NMAX:-7}"
HF_REPO="$(md qwen-hf-repo)"; HF_REPO="${HF_REPO:-unsloth/Qwen3.8-27B-GGUF}"
HF_FILE="$(md qwen-hf-file)"; HF_FILE="${HF_FILE:-Qwen3.8-27B-UD-Q4_K_XL.gguf}"
DRAFT_REPO="$(md qwen-draft-repo)"; DRAFT_REPO="${DRAFT_REPO:-incoai/Qwen3.8-27B-DFlash2-GGUF}"
DRAFT_FILE="$(md qwen-draft-file)"; DRAFT_FILE="${DRAFT_FILE:-Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
LCPP_PR="$(md qwen-lcpp-pr)"; LCPP_PR="${LCPP_PR:-27342}"
MMVQ_MAX="$(md qwen-mmvq-max)"; MMVQ_MAX="${MMVQ_MAX:-2}"

# Pass 1: turn ECC off and reboot. Frees ~1 GB VRAM, which this configuration needs.
if nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | grep -qi Enabled; then
  echo "[startup] disabling ECC + rebooting"; nvidia-smi -e 0 || true; reboot; exit 0
fi

echo "[startup] ECC off, provisioning in parallel $(date -u +%H:%M:%S)"
mkdir -p /opt/models && chmod 777 /opt/models
MODEL=/opt/models/model.gguf
DRAFT=/opt/models/dflash2.gguf

dl() {  # dl <repo> <file> <dest> <parts>
  local url="https://huggingface.co/$1/resolve/main/$2" dest="$3" n="${4:-8}"
  [ -f "$dest" ] && return 0
  local sz; sz=$(curl -sIL "$url" | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print $2}')
  echo "[startup] fetching $2 size=$sz"
  local ch=$(( sz / n )) i s e
  for i in $(seq 0 $((n-1))); do
    s=$(( i * ch ))
    if [ "$i" -eq $((n-1)) ]; then e=$(( sz - 1 )); else e=$(( s + ch - 1 )); fi
    curl -sL -r "$s-$e" "$url" -o "$dest.part.$i" &
  done
  wait
  cat "$dest".part.* > "$dest"; rm -f "$dest".part.*
  local act; act=$(stat -c%s "$dest")
  [ "$act" = "$sz" ] || { echo "[startup] SIZE MISMATCH $act != $sz for $2"; exit 1; }
}

# (a) build the patched llama.cpp
(
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq cmake build-essential ccache git
  export PATH=/usr/local/cuda-12.9/bin:/usr/local/cuda/bin:$PATH
  cd /opt
  if [ ! -d llama.cpp/.git ]; then
    git init llama.cpp
    git -C llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
  fi
  git -C llama.cpp fetch --depth 1 origin "pull/$LCPP_PR/head:pr$LCPP_PR"
  git -C llama.cpp checkout -f "pr$LCPP_PR"
  md qwen-patch > /opt/mmvq.patch
  git -C llama.cpp apply /opt/mmvq.patch
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
    -DGGML_CCACHE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
  cmake --build llama.cpp/build -j"$(nproc)" --target llama-server llama-quantize
  echo "[startup] build ready $(date -u +%H:%M:%S)"
) &
BPID=$!

# (b) target weights, (c) draft head
( dl "$HF_REPO" "$HF_FILE" "$MODEL" 8;        echo "[startup] target ready $(date -u +%H:%M:%S)" ) &
MPID=$!
( dl "$DRAFT_REPO" "$DRAFT_FILE" "$DRAFT" 4; echo "[startup] draft ready $(date -u +%H:%M:%S)" ) &
DPID=$!

wait $BPID; wait $MPID; wait $DPID

# The block drafter is used exactly as published. Grafting a cheaper output head onto it is
# measured and loses: it runs a top-k over those logits and traces a path through them, so the
# logit precision is structural for it in a way it is not for a head that only picks one token.
export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}

echo "[startup] launching server $(date -u +%H:%M:%S)"
cat > /etc/systemd/system/qwen-server.service <<UNIT
[Unit]
Description=llama.cpp server, Qwen3.8-27B UD-Q4_K_XL with DFlash2 block-drafting speculative decoding
After=network-online.target

[Service]
Environment=GGML_MMVQ_MAX=$MMVQ_MAX
Environment=GGML_MMVQ_MAX_Q6K=$MMVQ_MAX
Environment=GGML_MMVQ_MAX_IQ4XS=$MMVQ_MAX
Environment=LLAMA_SCHED_POOL=8
Environment=LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.9/lib64
ExecStart=/opt/llama.cpp/build/bin/llama-server \\
  --model $MODEL --alias Qwen3.8-27B-UD-Q4KXL-MTP \\
  --host 0.0.0.0 --port 8080 --jinja --tools all --metrics \\
  --parallel 1 --flash-attn on --fit on --fit-target $FIT_TARGET \\
  -ub $UB -b 2048 --cache-type-k $KV --cache-type-v $KV \\
  --no-mmap --threads 8 -bs \\
  --spec-type draft-dflash --spec-draft-model $DRAFT --spec-draft-ngl 99 \\
  --spec-draft-n-max $NMAX --spec-draft-n-min 1 --spec-draft-p-min 0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now qwen-server
echo "[startup] done $(date -u +%H:%M:%S)"
