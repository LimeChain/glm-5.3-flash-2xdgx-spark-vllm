#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
rank="${1:?rank required}"
hcas="${2:?HCA list required}"
cross_nic="${3:-0}"
[[ "$rank" == 0 || "$rank" == 1 ]] || exit 2
: "${HEAD_IP:?HEAD_IP required}"
: "${FABRIC_IF:?FABRIC_IF required}"
: "${GID_INDEX:?GID_INDEX required}"
: "${IMAGE:?IMAGE required}"
name="glm53-fabric-$rank-$$"
result_dir="$(mktemp -d)"
extra_env=()
[[ -z "${NCCL_PROTO_VALUE:-}" ]] || extra_env+=(-e "NCCL_PROTO=$NCCL_PROTO_VALUE")
[[ -z "${NCCL_CHANNELS:-}" ]] || extra_env+=(-e "NCCL_MAX_NCHANNELS=$NCCL_CHANNELS")
cleanup() {
  docker rm -f "$name" >/dev/null 2>&1 || true
  rm -rf -- "$result_dir"
}
trap cleanup EXIT
docker run --rm --name "$name" --gpus all --network host --ipc host \
  --device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -e GLM53_GB10_PATCH_DISABLE=1 -e "RANK=$rank" -e WORLD_SIZE=2 \
  -e "MASTER_ADDR=$HEAD_IP" -e MASTER_PORT=29561 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e "NCCL_IB_HCA=$hcas" \
  -e "NCCL_IB_GID_INDEX=$GID_INDEX" -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e "NCCL_SOCKET_IFNAME=$FABRIC_IF" -e "GLOO_SOCKET_IFNAME=$FABRIC_IF" \
  -e NCCL_IB_MERGE_NICS=0 -e "NCCL_CROSS_NIC=$cross_nic" \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=INFO \
  -e "BENCH_GRAPHS=${BENCH_GRAPHS:-0}" -e "BENCH_BACKEND=${BENCH_BACKEND:-torch}" "${extra_env[@]}" \
  -e BENCH_OUTPUT=/result/fabric.json -v "$result_dir:/result" \
  -v "$ROOT_DIR/bench:/bench:ro" --entrypoint python3 "$IMAGE" /bench/fabric.py
if [[ "$rank" == 0 ]]; then cat "$result_dir/fabric.json"; fi
