#!/usr/bin/env bash
# Stop (default) or delete the instance. On-demand L4 bills ~$0.81/hr whether idle or not.
#   bash scripts/teardown.sh          # stop, keeps the boot disk + downloaded model
#   bash scripts/teardown.sh delete   # full teardown, no further charges
set -euo pipefail
INSTANCE="${INSTANCE:-qwen38-27b-l4-od}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ACTION="${1:-stop}"

ZONE="$(gcloud compute instances list --project="$PROJECT" \
        --filter="name=$INSTANCE" --format='value(zone)')"
[ -n "$ZONE" ] || { echo "$INSTANCE not found in $PROJECT"; exit 1; }

case "$ACTION" in
  stop)   gcloud compute instances stop   "$INSTANCE" --zone="$ZONE" --project="$PROJECT" ;;
  delete) gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --quiet ;;
  *) echo "usage: $0 [stop|delete]"; exit 1 ;;
esac
