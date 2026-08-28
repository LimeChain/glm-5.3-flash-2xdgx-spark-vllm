#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
# shellcheck disable=SC1091
source "$ROOT_DIR/config/versions.env"
set +a

image="${IMAGE:-$DEFAULT_IMAGE}"
model="${MODEL:-cyijun2k/glm-5.3-flash-tiny-random-nvfp4}"
name="${NAME:-glm53-gb10-mock-smoke-$$}"
port="${PORT:-18000}"

docker_args=(
  --detach --name "$name"
  --gpus all
  --ipc=host
  --publish "$port:8000"
)
served_model="$model"
if [[ "$model" = /* ]]; then
  [[ -f "$model/config.json" ]] || {
    echo "missing $model/config.json" >&2
    exit 2
  }
  docker_args+=(--volume "$model:/model:ro")
  served_model=/model
fi
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  docker_args+=(--env "HF_ENDPOINT=$HF_ENDPOINT")
fi

serve_args=(
  "$served_model"
  --served-model-name glm-5.3-flash-tiny-random-nvfp4
  --attention-backend FLASHINFER_MLA_SPARSE_SM120
  --kv-cache-dtype fp8_ds_mla
  --block-size 256
  --moe-backend marlin
  --max-model-len 512
  --max-num-seqs 2
  --gpu-memory-utilization 0.10
  --enforce-eager
)
if [[ "${MTP:-0}" == 1 ]]; then
  serve_args+=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
fi

cleanup() {
  docker rm --force --volumes "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run "${docker_args[@]}" "$image" "${serve_args[@]}" >/dev/null

healthy=0
for _ in $(seq 1 180); do
  if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
    healthy=1
    break
  fi
  if [[ "$(docker inspect --format '{{.State.Running}}' "$name")" != true ]]; then
    echo "vLLM container exited before health check" >&2
    docker logs "$name" >&2
    exit 1
  fi
  sleep 2
done
[[ "$healthy" == 1 ]] || { echo "health check timed out" >&2; exit 1; }

curl --fail --silent --show-error \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-tiny-random-nvfp4","messages":[{"role":"user","content":"ping"}],"max_tokens":4,"temperature":0}'
