#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/config/cluster.env}"
VERSIONS_FILE="$ROOT_DIR/config/versions.env"

[[ -r "$CONFIG_FILE" ]] || { echo "missing config: $CONFIG_FILE (copy config/cluster.env.example first)" >&2; exit 2; }
[[ -r "$VERSIONS_FILE" ]] || { echo "missing versions file: $VERSIONS_FILE" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$VERSIONS_FILE"
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

require() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "missing required config: $name" >&2; exit 3; }
}
for name in IMAGE CONTAINER_NAME SERVED_MODEL_NAME MODEL_HOST CACHE_HOST HEAD_IP WORKER_IP FABRIC_IF FABRIC_HCA MASTER_PORT API_PORT MTP_TOKENS MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION; do
  require "$name"
done

RANK="${1:?usage: rank-tp2.sh <0|1>}"
case "$RANK" in 0|1) ;; *) echo "rank must be 0 or 1" >&2; exit 4 ;; esac
case "$HEAD_IP:$WORKER_IP" in *192.0.2.*|*CHANGE_ME*) echo "replace the documentation-only fabric addresses in $CONFIG_FILE" >&2; exit 5 ;; esac
case "$MTP_TOKENS" in 3) ;; *) echo "this release profile is qualified only with MTP_TOKENS=3" >&2; exit 6 ;; esac
[[ "$MAX_MODEL_LEN" == 262144 ]] || { echo "qualified MAX_MODEL_LEN is 262144" >&2; exit 7; }
[[ "$MAX_NUM_SEQS" == 12 ]] || { echo "qualified MAX_NUM_SEQS is 12" >&2; exit 8; }
[[ "$MAX_NUM_BATCHED_TOKENS" == 8192 ]] || { echo "qualified MAX_NUM_BATCHED_TOKENS is 8192" >&2; exit 9; }
[[ "$GPU_MEMORY_UTILIZATION" == 0.82 ]] || { echo "qualified GPU_MEMORY_UTILIZATION is 0.82" >&2; exit 10; }

[[ -d "$MODEL_HOST" ]] || { echo "missing model directory: $MODEL_HOST" >&2; exit 20; }
[[ -f "$MODEL_HOST/config.json" ]] || { echo "missing model config: $MODEL_HOST/config.json" >&2; exit 21; }
[[ -f "$MODEL_HOST/model.safetensors.index.json" ]] || { echo "missing model index: $MODEL_HOST/model.safetensors.index.json" >&2; exit 22; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "missing local image: $IMAGE" >&2; exit 23; }

read -r image_id image_revision local_heads base_digest < <(
  docker image inspect "$IMAGE" --format '{{.Id}} {{index .Config.Labels "org.opencontainers.image.revision"}} {{index .Config.Labels "io.glm53-gb10.tp-local-heads"}} {{index .Config.Labels "io.glm53-gb10.base-image-digest"}}'
)
[[ -n "$image_revision" && "$image_revision" != uncommitted ]] || { echo "image has no immutable source revision label" >&2; exit 24; }
[[ "$local_heads" == "32,64" ]] || { echo "image local-head contract mismatch: $local_heads" >&2; exit 25; }
[[ "$base_digest" == "${BASE_IMAGE##*@}" ]] || { echo "base-image digest label mismatch: $base_digest" >&2; exit 26; }

resolve_gid() {
  local ip="$1" a b c d hex gid_path index gid_type value
  IFS=. read -r a b c d <<<"$ip"
  printf -v hex '%02x%02x:%02x%02x' "$a" "$b" "$c" "$d"
  for gid_path in "/sys/class/infiniband/$FABRIC_HCA/ports/1/gids/"*; do
    [[ -e "$gid_path" ]] || continue
    index="${gid_path##*/}"
    gid_type="$(cat "/sys/class/infiniband/$FABRIC_HCA/ports/1/gid_attrs/types/$index" 2>/dev/null || true)"
    [[ "$gid_type" == "RoCE v2" ]] || continue
    value="$(cat "$gid_path" 2>/dev/null || true)"
    case "$value" in *ffff:"$hex") printf '%s' "$index"; return 0 ;; esac
  done
  return 1
}

if [[ "$RANK" == 0 ]]; then
  HOST_IP="$HEAD_IP"
  HEADLESS=()
else
  HOST_IP="$WORKER_IP"
  HEADLESS=(--headless)
fi
GID_INDEX="$(resolve_gid "$HOST_IP")" || { echo "cannot resolve RoCE v2 GID index for $HOST_IP on $FABRIC_HCA" >&2; exit 27; }

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  printf 'preflight_ok rank=%s image=%s revision=%s local_heads=%s base_digest=%s gid=%s model=%s\n' \
    "$RANK" "$image_id" "$image_revision" "$local_heads" "$base_digest" "$GID_INDEX" "$MODEL_HOST"
  exit 0
fi

if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "refusing to replace running container: $CONTAINER_NAME" >&2
  exit 30
fi
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
mkdir -p "$CACHE_HOST"

SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")

docker run -d \
  --name "$CONTAINER_NAME" --restart no --stop-timeout 60 \
  --gpus all --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST:/model:ro" \
  -v "$CACHE_HOST:/cache" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$FABRIC_HCA" -e NCCL_IB_GID_INDEX="$GID_INDEX" \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_SOCKET_IFNAME="$FABRIC_IF" -e GLOO_SOCKET_IFNAME="$FABRIC_IF" -e TP_SOCKET_IFNAME="$FABRIC_IF" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    /model \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 --port "$API_PORT" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
    --kv-cache-dtype fp8_ds_mla --block-size 256 \
    --moe-backend marlin \
    --enforce-eager \
    --skip-mm-profiling \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --generation-config vllm \
    "${SPEC_ARGS[@]}" \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$RANK" \
    --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
    "${HEADLESS[@]}"

sleep 2
docker inspect "$CONTAINER_NAME" --format 'running={{.State.Running}} restart_count={{.RestartCount}} oom_killed={{.State.OOMKilled}}'
