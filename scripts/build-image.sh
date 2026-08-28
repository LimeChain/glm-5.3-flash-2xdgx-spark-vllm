#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
# shellcheck disable=SC1091
source "$ROOT_DIR/config/versions.env"
set +a

revision="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf uncommitted)"
image="${IMAGE:-$DEFAULT_IMAGE}"

docker build \
  --platform linux/arm64 \
  --file "$ROOT_DIR/container/Dockerfile" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "REVISION=$revision" \
  --build-arg "BASE_IMAGE_DIGEST=${BASE_IMAGE##*@}" \
  --tag "$image" \
  "$ROOT_DIR"

docker image inspect "$image" --format 'image={{.Id}}'
