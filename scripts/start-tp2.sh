#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/config/cluster.env}"
[[ -r "$CONFIG_FILE" ]] || { echo "missing config: $CONFIG_FILE" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

: "${WORKER_SSH:?missing WORKER_SSH}"
: "${REMOTE_ROOT:?missing REMOTE_ROOT}"
: "${CONTAINER_NAME:?missing CONTAINER_NAME}"
: "${API_PORT:?missing API_PORT}"

printf -v remote_config_q '%q' "$REMOTE_ROOT/config/cluster.env"
printf -v remote_rank_q '%q' "$REMOTE_ROOT/scripts/rank-tp2.sh"

"$ROOT_DIR/scripts/preflight-tp2.sh"

if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "container already running on head: $CONTAINER_NAME" >&2
  exit 10
fi
if ssh -o BatchMode=yes "$WORKER_SSH" "docker ps --format '{{.Names}}' | grep -Fxq '$CONTAINER_NAME'"; then
  echo "container already running on worker: $CONTAINER_NAME" >&2
  exit 11
fi

echo "Starting worker rank..."
ssh -o BatchMode=yes "$WORKER_SSH" "CONFIG_FILE=$remote_config_q $remote_rank_q 1"
sleep 8

echo "Starting head rank..."
"$ROOT_DIR/scripts/rank-tp2.sh" 0

start_epoch="$(date +%s)"
for _ in $(seq 1 240); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    elapsed="$(( $(date +%s) - start_epoch ))"
    echo "GLM_READY elapsed_seconds=$elapsed endpoint=http://127.0.0.1:$API_PORT"
    curl -fsS --max-time 10 "http://127.0.0.1:$API_PORT/v1/models"
    exit 0
  fi
  head_state="$(docker inspect "$CONTAINER_NAME" --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}' 2>/dev/null || echo missing)"
  worker_state="$(ssh -o BatchMode=yes "$WORKER_SSH" "docker inspect '$CONTAINER_NAME' --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}' 2>/dev/null || echo missing")"
  if [[ "$head_state" != true\|* || "$worker_state" != true\|* ]]; then
    echo "rank failure during startup: head=$head_state worker=$worker_state" >&2
    echo "--- head logs ---" >&2
    docker logs --tail 160 "$CONTAINER_NAME" >&2 || true
    echo "--- worker logs ---" >&2
    ssh -o BatchMode=yes "$WORKER_SSH" "docker logs --tail 160 '$CONTAINER_NAME'" >&2 || true
    exit 12
  fi
  sleep 15
done

echo "timeout waiting for GLM health" >&2
exit 13
