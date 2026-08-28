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
: "${CONTAINER_NAME:?missing CONTAINER_NAME}"

stop_local() {
  docker stop --time 30 "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

stop_local
ssh -o BatchMode=yes "$WORKER_SSH" "docker stop --time 30 '$CONTAINER_NAME' >/dev/null 2>&1 || true; docker rm '$CONTAINER_NAME' >/dev/null 2>&1 || true"

head_left="$(docker ps -a --format '{{.Names}}' | grep -Fx "$CONTAINER_NAME" || true)"
worker_left="$(ssh -o BatchMode=yes "$WORKER_SSH" "docker ps -a --format '{{.Names}}' | grep -Fx '$CONTAINER_NAME' || true")"
[[ -z "$head_left$worker_left" ]] || { echo "container remains: head=$head_left worker=$worker_left" >&2; exit 10; }
echo "GLM_STOPPED on both ranks"
