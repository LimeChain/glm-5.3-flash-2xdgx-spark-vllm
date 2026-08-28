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

printf -v remote_config_q '%q' "$REMOTE_ROOT/config/cluster.env"
printf -v remote_rank_q '%q' "$REMOTE_ROOT/scripts/rank-tp2.sh"

"$ROOT_DIR/scripts/rank-tp2.sh" 0
ssh -o BatchMode=yes "$WORKER_SSH" "test -x $remote_rank_q && test -r $remote_config_q"
ssh -o BatchMode=yes "$WORKER_SSH" "CONFIG_FILE=$remote_config_q PREFLIGHT_ONLY=1 $remote_rank_q 1"
echo "TP2_PREFLIGHT_OK"
