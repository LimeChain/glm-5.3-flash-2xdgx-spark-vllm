# Two-node DGX Spark deployment

This guide reproduces the qualified TP2 topology without carrying private hostnames, IP addresses, paths, or credentials into the repository.

## Safety boundary

- The API binds to `127.0.0.1` on the head node.
- Model files are mounted read-only.
- Hugging Face and Transformers are forced offline at serve time.
- The launcher refuses to replace an already-running container.
- The qualified configuration is fail-closed: MTP3, 262,144 context, C12 admission, 8,192 batched tokens, 0.82 GPU-memory utilization, eager execution, and thinking enabled at High.
- `stop-tp2.sh` stops and removes only the configured container name on the two configured nodes.

## 1. Fabric

Provide a routable RoCE v2 address on each DGX Spark and record:

- head and worker fabric IPv4 addresses;
- Linux interface name used by NCCL/Gloo;
- InfiniBand/RoCE HCA name exposed under `/sys/class/infiniband`.

The rank launcher resolves the RoCE v2 GID index matching each configured address. It fails rather than guessing.

## 2. Source and image parity

Use the same checked-out commit on both nodes. Build the image on each node with the same tag:

```bash
IMAGE=glm53-sm121:local ./scripts/build-image.sh
```

The preflight requires:

- a non-`uncommitted` OCI source-revision label;
- `io.glm53-gb10.tp-local-heads=32,64`;
- the expected immutable base-image digest;
- the model config and SafeTensors index.

For stronger deployment control, record `docker image inspect --format '{{.Id}}' glm53-sm121:local` on each node and require the IDs to match before starting.

## 3. Checkpoint parity

Download `LibertAIDAI/GLM-5.3-Flash-NVFP4` at revision:

```text
9e0d74e3cef17f634e84fb8e2223707e02616290
```

Place it at the same path on both systems. The public launcher verifies required files but intentionally does not embed a machine-specific local checkpoint manifest. Operators who need strict file-byte authority should generate and compare a complete sorted SHA-256 manifest on both nodes before launch.

## 4. Configuration

On both nodes:

```bash
cp config/cluster.env.example config/cluster.env
```

Edit all environment-specific values. `192.0.2.0/24` is a documentation-only address range; preflight rejects the example addresses.

The head needs batch-mode SSH access to `WORKER_SSH`. `REMOTE_ROOT` must point to the repository on the worker.

Do not add API keys, SSH keys, Hugging Face tokens, or registry credentials to `cluster.env`.

## 5. Preflight

From the head:

```bash
./scripts/preflight-tp2.sh
```

Both ranks must print `preflight_ok`, followed by `TP2_PREFLIGHT_OK`.

## 6. Start and observe

```bash
./scripts/start-tp2.sh
```

The worker starts first. The head then starts and polls `/health` for up to one hour while checking both container states. A rank exit or OOM terminates startup with logs rather than silently waiting.

After `GLM_READY`:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

The server defaults to `enable_thinking=true` and `reasoning_effort=high`. With the `glm45` reasoning parser, OpenAI-compatible responses expose reasoning separately from final content.

Keep the service private behind SSH forwarding, Tailscale, or another authenticated transport if remote clients need access. Do not expose an unauthenticated vLLM endpoint directly to the public internet.

## 7. Benchmark

Run only against a drained endpoint:

```bash
python3 bench/benchmark.py \
  --base http://127.0.0.1:8000 \
  --label cold-1 \
  --output results/cold-1.json
```

Cold-start again and repeat with a different output path. Do not combine results from different model revisions, images, speculation settings, context settings, or workloads into one table.

## 8. Stop

```bash
./scripts/stop-tp2.sh
```

Verify no configured containers remain on either node before changing images or fabric settings.
