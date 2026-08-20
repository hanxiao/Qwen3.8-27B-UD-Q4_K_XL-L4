#!/usr/bin/env bash
# Provision an ON-DEMAND NVIDIA L4 (g2-standard-8) on GCP serving Qwen3.8-27B
# (Unsloth UD-Q4_K_XL GGUF) with MTP speculative decoding, and block until ready.
#
# Design notes (learned the hard way, see README):
#  - ON-DEMAND, not spot. Spot L4s get preempted mid-benchmark and lose their IP.
#  - The instance builds llama.cpp from source, because the one change that matters
#    (routing speculative verification to the MMQ kernel) is a source patch. The build
#    runs in parallel with the two model downloads; it is the long pole, ~20 min on
#    8 vCPUs. Use scripts/build-llamacpp.sh on a bigger CPU box and copy bin/ across if
#    you would rather not wait.
#  - Models are fetched with ranged `curl`, NOT huggingface_hub. The GCP deep-learning
#    images do not ship `pip` on PATH; an hf_hub download step dies with exit 127
#    before it ever starts.
#  - The ECC-off reboot is load-bearing. It frees ~1 GB of VRAM.
#  - Zones are enumerated live. On-demand L4 capacity is scarce; expect a long walk
#    through STOCKOUT zones before one lands. That is the script working, not failing.
#
# Usage:  bash scripts/provision-ondemand.sh
# Env:    INSTANCE, MACHINE, PROJECT, ZONES, FIT_TARGET, KV, NMAX, MMVQ_MAX, INTERNAL, DISK
set -euo pipefail

INSTANCE="${INSTANCE:-qwen38-27b-l4-od}"
MACHINE="${MACHINE:-g2-standard-8}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
# The context size is planned on the instance by llama.cpp --fit; what is set here is
# the margin it must leave free and the KV cache type, which together decide the window.
# Boot disk in GB. 120 serves the model; capture work for a draft head needs far more,
# at about 10.3 GB per million tokens of hidden state.
DISK="${DISK:-120}"
FIT_TARGET="${FIT_TARGET:-768}"
KV="${KV:-q4_0}"
NMAX="${NMAX:-7}"
UB="${UB:-256}"
MMVQ_MAX="${MMVQ_MAX:-2}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

[ -n "$PROJECT" ] || { echo "No GCP project set. Use: gcloud config set project <id>"; exit 1; }

if [ -z "${ZONES:-}" ]; then
  ALL="$(gcloud compute machine-types list --filter="name=$MACHINE" --format='value(zone)' --project="$PROJECT")"
  # US zones first, then the rest. `|| true` matters: a grep that matches nothing exits
  # 1 and would take the whole script down under `set -e`.
  ZONES="$( { printf '%s\n' "$ALL" | grep '^us-' || true; printf '%s\n' "$ALL" | grep -v '^us-' || true; } )"
fi

say(){ printf '\n\033[1;36m> %s\033[0m\n' "$*"; }

INTERNAL="${INTERNAL:-1}"
if [ "$INTERNAL" = "1" ]; then
  NET_TAG=llama-internal
  say "Ensuring VPC-only firewall for :8080 ..."
  gcloud compute firewall-rules create allow-llama-internal-8080 \
    --allow=tcp:8080 --target-tags=llama-internal --source-ranges=10.128.0.0/9 \
    --project="$PROJECT" 2>/dev/null || true
else
  NET_TAG=llama-server
  say "Ensuring PUBLIC firewall for :8080 (no authentication, benchmark use only) ..."
  gcloud compute firewall-rules create allow-llama-8080 \
    --allow=tcp:8080 --target-tags=llama-server --source-ranges=0.0.0.0/0 \
    --project="$PROJECT" 2>/dev/null || true
fi

META="startup-script=$HERE/scripts/startup.sh,qwen-patch=$HERE/patches/0001-mmvq-runtime-crossover.patch"

ZONE=""
for z in $ZONES; do
  say "Requesting ON-DEMAND $MACHINE (1x L4) in $z ..."
  if gcloud compute instances create "$INSTANCE" \
      --project="$PROJECT" \
      --zone="$z" --machine-type="$MACHINE" \
      --maintenance-policy=TERMINATE \
      --image-family=common-cu129-ubuntu-2204-nvidia-580 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=${DISK}GB --boot-disk-type=pd-ssd \
      --tags="$NET_TAG" \
      --metadata="qwen-fit-target=$FIT_TARGET,qwen-kv=$KV,qwen-nmax=$NMAX,qwen-ub=$UB,qwen-mmvq-max=$MMVQ_MAX" \
      --metadata-from-file="$META" 2>&1 | tail -3; then
    ZONE="$z"; break
  fi
  say "No capacity in $z, trying next ..."
done
[ -n "$ZONE" ] || { say "No on-demand L4 capacity in any offering zone"; exit 1; }

IP="$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
INTERNAL_IP="$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
        --format='value(networkInterfaces[0].networkIP)')"
echo "ZONE=$ZONE"
echo "IP=$IP"
say "Created in $ZONE (IP $IP). Waiting for /health (ECC reboot + build + 19.6 GB pull) ..."
probe() {
  if [ "$INTERNAL" = "1" ]; then
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
      --command 'curl -fsS --max-time 5 http://127.0.0.1:8080/health' 2>/dev/null
  else
    curl -fsS --max-time 5 "http://$IP:8080/health" 2>/dev/null
  fi
}
for i in $(seq 1 300); do
  if probe | grep -q '"status":"ok"'; then
    say "READY"
    cat <<EOM
  Internal: http://$INTERNAL_IP:8080  (VPC only when INTERNAL=1)
  API:      http://$INTERNAL_IP:8080/v1/chat/completions
  Metrics:  http://$INTERNAL_IP:8080/metrics
  Bench:    gcloud compute scp scripts/bench.sh $INSTANCE:~/bench.sh --zone=$ZONE --project=$PROJECT
            gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT --command 'bash ~/bench.sh 127.0.0.1:8080'
  Logs:     gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT --command 'sudo tail -50 /var/log/qwen-startup.log'
  Stop:     gcloud compute instances stop   $INSTANCE --zone=$ZONE --project=$PROJECT
  Delete:   gcloud compute instances delete $INSTANCE --zone=$ZONE --project=$PROJECT
EOM
    exit 0
  fi
  printf '.'; sleep 10
done

say "TIMED OUT. Inspect the boot log:"
echo "  gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT --command 'sudo tail -80 /var/log/qwen-startup.log'"
exit 1
