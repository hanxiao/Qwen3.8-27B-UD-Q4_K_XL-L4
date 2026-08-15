#!/usr/bin/env bash
# Provision an ON-DEMAND NVIDIA L4 (g2-standard-8) on GCP serving Qwen3.8-27B
# (Unsloth UD-Q4_K_XL GGUF) with MTP speculative decoding, and block until ready.
#
# Design notes (learned the hard way, see README "Pitfalls"):
#  - ON-DEMAND, not spot. Spot L4s get preempted mid-benchmark and lose their IP.
#  - The model is fetched with an 8-way ranged `curl`, NOT huggingface_hub. The GCP
#    deep-learning images do not ship `pip` on PATH; an hf_hub download step dies with
#    exit 127 before it ever starts. Ranged curl needs nothing but curl and is faster
#    (~100 s for 17.9 GB) than a single-stream hf download anyway.
#  - The ECC-off reboot is load-bearing. It frees ~1 GB of VRAM. Without it the
#    17.9 GB of weights plus a 104192-token KV cache do not fit in 24 GB.
#  - Zones are enumerated live. On-demand L4 capacity is scarce; expect a long walk
#    through STOCKOUT zones before one lands. That is the script working, not failing.
#
# Usage:  bash scripts/provision-ondemand.sh
# Env:    INSTANCE, MACHINE, PROJECT, ZONES, HF_REPO, HF_FILE, CTX
set -euo pipefail

INSTANCE="${INSTANCE:-qwen38-27b-l4-od}"
MACHINE="${MACHINE:-g2-standard-8}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
HF_REPO="${HF_REPO:-unsloth/Qwen3.8-27B-GGUF}"
HF_FILE="${HF_FILE:-Qwen3.8-27B-UD-Q4_K_XL.gguf}"
CTX="${CTX:-104192}"
DL_PARTS="${DL_PARTS:-8}"

[ -n "$PROJECT" ] || { echo "No GCP project set. Use: gcloud config set project <id>"; exit 1; }

if [ -z "${ZONES:-}" ]; then
  ALL="$(gcloud compute machine-types list --filter="name=$MACHINE" --format='value(zone)' --project="$PROJECT")"
  # US zones first, then the rest. `|| true` matters: a grep that matches nothing exits
  # 1 and would take the whole script down under `set -e`.
  ZONES="$( { printf '%s\n' "$ALL" | grep '^us-' || true; printf '%s\n' "$ALL" | grep -v '^us-' || true; } )"
fi

say(){ printf '\n\033[1;36m> %s\033[0m\n' "$*"; }

STARTUP="$(mktemp)"; cat > "$STARTUP" <<SH
#!/bin/bash
set -e
exec >>/var/log/qwen-startup.log 2>&1
echo "[startup] boot \$(date -u +%H:%M:%S)"

# Pass 1: turn ECC off and reboot. Frees ~1 GB VRAM, which is what makes ctx=$CTX fit.
if nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | grep -qi Enabled; then
  echo "[startup] disabling ECC + rebooting"; nvidia-smi -e 0 || true; reboot; exit 0
fi

echo "[startup] ECC off - parallel provision \$(date -u +%H:%M:%S)"
mkdir -p /opt/models
MODEL=/opt/models/model.gguf

# (a) docker engine + llama.cpp server image
(
  if ! command -v docker >/dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
  fi
  docker image inspect ghcr.io/ggml-org/llama.cpp:server-cuda >/dev/null 2>&1 \
    || docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
  echo "[startup] docker+image ready \$(date -u +%H:%M:%S)"
) &
DPID=\$!

# (b) the 17.9 GB GGUF via $DL_PARTS-way ranged curl. No pip, no hf_hub.
(
  if [ ! -f "\$MODEL" ]; then
    URL="https://huggingface.co/$HF_REPO/resolve/main/$HF_FILE"
    SZ=\$(curl -sIL "\$URL" | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print \$2}')
    echo "[startup] model size=\$SZ"
    N=$DL_PARTS
    CH=\$(( SZ / N ))
    for i in \$(seq 0 \$((N-1))); do
      S=\$(( i * CH ))
      if [ \$i -eq \$((N-1)) ]; then E=\$(( SZ - 1 )); else E=\$(( S + CH - 1 )); fi
      curl -sL -r "\$S-\$E" "\$URL" -o /opt/models/part.\$i &
    done
    wait
    cat /opt/models/part.* > "\$MODEL"
    rm -f /opt/models/part.*
    ACT=\$(stat -c%s "\$MODEL")
    [ "\$ACT" = "\$SZ" ] || { echo "[startup] SIZE MISMATCH \$ACT != \$SZ"; exit 1; }
    echo "[startup] model ready \$(date -u +%H:%M:%S)"
  fi
) &
MPID=\$!

wait \$DPID; wait \$MPID

echo "[startup] launching server \$(date -u +%H:%M:%S)"
docker rm -f llama-server 2>/dev/null || true
docker run -d --name llama-server --restart unless-stopped --gpus all -p 8080:8080 \
  -v /opt/models:/models ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model /models/model.gguf \
  --alias Qwen3.8-27B-UD-Q4KXL-MTP --host 0.0.0.0 --port 8080 --jinja --tools all \
  --ctx-size $CTX --parallel 1 --flash-attn on -ngl 99 -ub 64 -b 512 \
  --no-mmap --threads 8 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.4 --no-warmup --metrics
echo "[startup] done \$(date -u +%H:%M:%S)"
SH

say "Ensuring firewall for :8080 ..."
gcloud compute firewall-rules create allow-llama-8080 \
  --allow=tcp:8080 --target-tags=llama-server --source-ranges=0.0.0.0/0 \
  --project="$PROJECT" 2>/dev/null || true

ZONE=""
for z in $ZONES; do
  say "Requesting ON-DEMAND $MACHINE (1x L4) in $z ..."
  if gcloud compute instances create "$INSTANCE" \
      --project="$PROJECT" \
      --zone="$z" --machine-type="$MACHINE" \
      --maintenance-policy=TERMINATE \
      --image-family=common-cu129-ubuntu-2204-nvidia-580 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=80GB --boot-disk-type=pd-ssd \
      --tags=llama-server \
      --metadata-from-file=startup-script="$STARTUP" 2>&1 | tail -3; then
    ZONE="$z"; break
  fi
  say "No capacity in $z, trying next ..."
done
rm -f "$STARTUP"
[ -n "$ZONE" ] || { say "No on-demand L4 capacity in any offering zone"; exit 1; }

# The instance is recreated as zones are retried, so always re-read the IP here.
IP="$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
echo "ZONE=$ZONE"
echo "IP=$IP"
say "Created in $ZONE (IP $IP). Waiting for /health (ECC reboot + 17.9 GB pull + load) ..."
for i in $(seq 1 180); do
  if curl -fsS --max-time 5 "http://$IP:8080/health" 2>/dev/null | grep -q '"status":"ok"'; then
    say "READY"
    cat <<EOF
  Web UI:   http://$IP:8080
  API:      http://$IP:8080/v1/chat/completions
  Metrics:  http://$IP:8080/metrics
  Bench:    gcloud compute scp scripts/bench.sh $INSTANCE:~/bench.sh --zone=$ZONE --project=$PROJECT
            gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT --command 'bash ~/bench.sh 127.0.0.1:8080'
  Stop:     gcloud compute instances stop   $INSTANCE --zone=$ZONE --project=$PROJECT
  Delete:   gcloud compute instances delete $INSTANCE --zone=$ZONE --project=$PROJECT
EOF
    exit 0
  fi
  printf '.'; sleep 10
done

say "TIMED OUT. Inspect the boot log:"
echo "  gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT --command 'sudo tail -50 /var/log/qwen-startup.log'"
exit 1
