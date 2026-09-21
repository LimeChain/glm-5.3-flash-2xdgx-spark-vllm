#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/config/cluster.env}"
VERSIONS_FILE="$ROOT_DIR/config/versions.env"
EXTRA_VLLM_ARGS=()

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
PROFILE="${PROFILE:-qualified}"
EXECUTION_MODE="${EXECUTION_MODE:-eager}"
REASONING_PARSER="${REASONING_PARSER:-glm45}"
case "$PROFILE" in qualified|experimental) ;; *) echo "unknown PROFILE=$PROFILE" >&2; exit 6 ;; esac
if [[ "$PROFILE" == qualified ]]; then
case "$MTP_TOKENS" in 3) ;; *) echo "this release profile is qualified only with MTP_TOKENS=3" >&2; exit 6 ;; esac
[[ "$MAX_MODEL_LEN" == 262144 ]] || { echo "qualified MAX_MODEL_LEN is 262144" >&2; exit 7; }
[[ "$MAX_NUM_SEQS" == 12 ]] || { echo "qualified MAX_NUM_SEQS is 12" >&2; exit 8; }
[[ "$MAX_NUM_BATCHED_TOKENS" == 8192 ]] || { echo "qualified MAX_NUM_BATCHED_TOKENS is 8192" >&2; exit 9; }
[[ "$GPU_MEMORY_UTILIZATION" == 0.82 ]] || { echo "qualified GPU_MEMORY_UTILIZATION is 0.82" >&2; exit 10; }
[[ "$EXECUTION_MODE" == eager && "$REASONING_PARSER" == glm45 ]] || { echo "qualified execution/parser changed" >&2; exit 10; }
[[ ${#EXTRA_VLLM_ARGS[@]} == 0 && -z "${CPUSET_CPUS:-}" && -z "${CHAT_TEMPLATE_HOST:-}" ]] || { echo "extra engine arguments, a custom template, or CPU affinity require PROFILE=experimental" >&2; exit 10; }
else
  [[ "$MTP_TOKENS" =~ ^[0-5]$ ]] || { echo "experimental MTP_TOKENS must be 0..5" >&2; exit 6; }
  echo "EXPERIMENTAL profile: new hardware qualification required" >&2
fi
EXECUTION_ARGS=()
CPU_ARGS=()
[[ -z "${CPUSET_CPUS:-}" ]] || CPU_ARGS=(--cpuset-cpus "$CPUSET_CPUS")
TEMPLATE_MOUNT=()
TEMPLATE_ARGS=()
template_hash=checkpoint
if [[ -n "${CHAT_TEMPLATE_HOST:-}" ]]; then
  [[ -f "$CHAT_TEMPLATE_HOST" ]] || { echo "missing chat template: $CHAT_TEMPLATE_HOST" >&2; exit 10; }
  template_hash="$(sha256sum "$CHAT_TEMPLATE_HOST" | cut -d ' ' -f1)"
  TEMPLATE_MOUNT=(-v "$CHAT_TEMPLATE_HOST:/recipe-chat-template.jinja:ro")
  TEMPLATE_ARGS=(--chat-template /recipe-chat-template.jinja)
fi
NCCL_TUNING_ARGS=()
[[ -z "${NCCL_PROTO_VALUE:-}" ]] || NCCL_TUNING_ARGS+=(-e "NCCL_PROTO=$NCCL_PROTO_VALUE")
[[ -z "${NCCL_CHANNELS:-}" ]] || NCCL_TUNING_ARGS+=(-e "NCCL_MAX_NCHANNELS=$NCCL_CHANNELS")
case "$EXECUTION_MODE" in eager) EXECUTION_ARGS=(--enforce-eager) ;; graph) ;; *) echo "unknown execution mode" >&2; exit 10 ;; esac

[[ -d "$MODEL_HOST" ]] || { echo "missing model directory: $MODEL_HOST" >&2; exit 20; }
[[ -f "$MODEL_HOST/config.json" ]] || { echo "missing model config: $MODEL_HOST/config.json" >&2; exit 21; }
[[ -f "$MODEL_HOST/model.safetensors.index.json" ]] || { echo "missing model index: $MODEL_HOST/model.safetensors.index.json" >&2; exit 22; }
python3 - "$MODEL_HOST" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
index = json.loads((p / 'model.safetensors.index.json').read_text())
missing = [name for name in set(index['weight_map'].values()) if not (p / name).is_file()]
if missing:
    raise SystemExit(f'missing model shards: {missing}')
PY
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
  profile_hash="$(printf '%s\n' "$PROFILE" "$MTP_TOKENS" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" "$GPU_MEMORY_UTILIZATION" "$EXECUTION_MODE" "$REASONING_PARSER" "$template_hash" "${CPUSET_CPUS:-}" "${NCCL_CHANNELS:-}" "${NCCL_PROTO_VALUE:-}" "${NCCL_CROSS_NIC_VALUE:-0}" "${EXTRA_VLLM_ARGS[@]-}" | sha256sum | cut -d ' ' -f1)"
  model_meta="$(cd "$MODEL_HOST" && sha256sum config.json model.safetensors.index.json chat_template.jinja tokenizer_config.json | sha256sum | cut -d ' ' -f1)"
  printf 'preflight_ok rank=%s image=%s revision=%s local_heads=%s base_digest=%s gid=%s profile=%s model_meta=%s model=%s\n' \
    "$RANK" "$image_id" "$image_revision" "$local_heads" "$base_digest" "$GID_INDEX" "$profile_hash" "$model_meta" "$MODEL_HOST"
  exit 0
fi

if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "refusing to replace running container: $CONTAINER_NAME" >&2
  exit 30
fi
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
mkdir -p "$CACHE_HOST"

SPEC_ARGS=()
if [[ "$MTP_TOKENS" != 0 ]]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
fi

docker run -d \
  --name "$CONTAINER_NAME" --restart no --stop-timeout 60 \
  --gpus all --network host --ipc host --shm-size 64g \
  "${CPU_ARGS[@]}" \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST:/model:ro" \
  -v "$CACHE_HOST:/cache" \
  "${TEMPLATE_MOUNT[@]}" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/cache/huggingface \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer -e CUDA_CACHE_PATH=/cache/cuda \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="${NCCL_HCA_LIST:-$FABRIC_HCA}" -e NCCL_IB_GID_INDEX="$GID_INDEX" \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_SOCKET_IFNAME="$FABRIC_IF" -e GLOO_SOCKET_IFNAME="$FABRIC_IF" -e TP_SOCKET_IFNAME="$FABRIC_IF" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC="${NCCL_CROSS_NIC_VALUE:-0}" -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG="${NCCL_LOG_LEVEL:-WARN}" \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "${NCCL_TUNING_ARGS[@]}" \
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
    "${EXECUTION_ARGS[@]}" \
    --skip-mm-profiling \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser "$REASONING_PARSER" \
    "${TEMPLATE_ARGS[@]}" \
    --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"high"}' \
    --generation-config vllm \
    "${SPEC_ARGS[@]}" \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$RANK" \
    --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
    "${HEADLESS[@]}" \
    "${EXTRA_VLLM_ARGS[@]}"

sleep 2
docker inspect "$CONTAINER_NAME" --format 'running={{.State.Running}} restart_count={{.RestartCount}} oom_killed={{.State.OOMKilled}}'
