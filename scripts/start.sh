#!/usr/bin/env bash
# Restart a stopped instance. `gcloud compute instances start` fails with STOCKOUT when
# the zone has no L4 free, and a stopped instance cannot move zones, so retry until one
# frees up.
#   bash scripts/start.sh            # retry for ~40 min
#   TRIES=200 bash scripts/start.sh
set -uo pipefail
INSTANCE="${INSTANCE:-qwen38-27b-l4-od}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
TRIES="${TRIES:-120}"
SLEEP="${SLEEP:-20}"

ZONE="${ZONE:-$(gcloud compute instances list --project="$PROJECT" \
        --filter="name=$INSTANCE" --format='value(zone)')}"
[ -n "$ZONE" ] || { echo "$INSTANCE not found in $PROJECT"; exit 1; }

for i in $(seq 1 "$TRIES"); do
  echo "attempt $i in $ZONE"
  gcloud compute instances start "$INSTANCE" --zone="$ZONE" --project="$PROJECT" 2>&1 | tail -2
  if [ "$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
        --format='value(status)')" = RUNNING ]; then
    IP="$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
          --format='value(networkInterfaces[0].networkIP)')"
    echo "RUNNING, internal IP $IP"
    for j in $(seq 1 90); do
      gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
        --command 'curl -fsS --max-time 5 http://127.0.0.1:8080/health' 2>/dev/null \
        | grep -q ok && { echo "READY http://$IP:8080"; exit 0; }
      sleep 10
    done
    echo "started but /health never came up; check /var/log/qwen-startup.log"; exit 1
  fi
  sleep "$SLEEP"
done
echo "no capacity in $ZONE after $TRIES attempts"; exit 1
