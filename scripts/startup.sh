#!/bin/bash
# Instance startup script. Reads its configuration from instance metadata so that this
# file needs no shell interpolation from the provisioning script.
set -e
exec >>/var/log/qwen-startup.log 2>&1
echo "[startup] boot $(date -u +%H:%M:%S)"

md() { curl -s -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null; }

CTX="$(md qwen-ctx)";        CTX="${CTX:-65536}"
NMAX="$(md qwen-nmax)";      NMAX="${NMAX:-5}"
HF_REPO="$(md qwen-hf-repo)"; HF_REPO="${HF_REPO:-unsloth/Qwen3.8-27B-GGUF}"
HF_FILE="$(md qwen-hf-file)"; HF_FILE="${HF_FILE:-Qwen3.8-27B-UD-Q4_K_XL.gguf}"
DRAFT_REPO="$(md qwen-draft-repo)"; DRAFT_REPO="${DRAFT_REPO:-ggml-org/Qwen3.8-27B-GGUF}"
DRAFT_FILE="$(md qwen-draft-file)"; DRAFT_FILE="${DRAFT_FILE:-mtp-Qwen3.8-27B-Q4_0.gguf}"
LCPP_PR="$(md qwen-lcpp-pr)"; LCPP_PR="${LCPP_PR:-27173}"
MMVQ_MAX="$(md qwen-mmvq-max)"; MMVQ_MAX="${MMVQ_MAX:-2}"
CHAIN_SUB="$(md qwen-chain-sub)"; CHAIN_SUB="${CHAIN_SUB:-98304}"

# Pass 1: turn ECC off and reboot. Frees ~1 GB VRAM, which this configuration needs.
if nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | grep -qi Enabled; then
  echo "[startup] disabling ECC + rebooting"; nvidia-smi -e 0 || true; reboot; exit 0
fi

echo "[startup] ECC off, provisioning in parallel $(date -u +%H:%M:%S)"
mkdir -p /opt/models && chmod 777 /opt/models
MODEL=/opt/models/model.gguf
DRAFT_RAW=/opt/models/mtp-raw.gguf
DRAFT=/opt/models/mtp-o2k.gguf

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
( dl "$DRAFT_REPO" "$DRAFT_FILE" "$DRAFT_RAW" 4; echo "[startup] draft ready $(date -u +%H:%M:%S)" ) &
DPID=$!

wait $BPID; wait $MPID; wait $DPID

# Retype the draft head's output tensor to Q2_K. Draft precision is not output precision:
# the target verifies every drafted token, so this can only change acceptance rate.
export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
if [ ! -f "$DRAFT" ]; then
  /opt/llama.cpp/build/bin/llama-quantize --allow-requantize \
    --output-tensor-type q2_K "$DRAFT_RAW" "$DRAFT" Q4_0
fi

echo "[startup] launching server $(date -u +%H:%M:%S)"
cat > /etc/systemd/system/qwen-server.service <<UNIT
[Unit]
Description=llama.cpp server, Qwen3.8-27B UD-Q4_K_XL with MTP speculative decoding
After=network-online.target

[Service]
Environment=GGML_MMVQ_MAX=$MMVQ_MAX
Environment=LLAMA_SPEC_CHAIN=1
Environment=LLAMA_SPEC_CHAIN_SUB=$CHAIN_SUB
Environment=LLAMA_SCHED_POOL=8
Environment=LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.9/lib64
ExecStart=/opt/llama.cpp/build/bin/llama-server \\
  --model $MODEL --alias Qwen3.8-27B-UD-Q4KXL-MTP \\
  --host 0.0.0.0 --port 8080 --jinja --tools all --metrics \\
  --ctx-size $CTX --parallel 1 --flash-attn on -ngl 99 -ub 512 -b 512 \\
  --no-mmap --threads 8 -bs \\
  --spec-type draft-mtp --spec-draft-model $DRAFT --spec-draft-ngl 99 \\
  --spec-draft-n-max $NMAX --spec-draft-n-min $NMAX --spec-draft-p-min 0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now qwen-server
echo "[startup] done $(date -u +%H:%M:%S)"
