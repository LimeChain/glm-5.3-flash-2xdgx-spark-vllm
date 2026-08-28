# GLM-5.3 Flash NVFP4 on 2× NVIDIA DGX Spark

Run the full **GLM-5.3 Flash NVFP4** checkpoint across two NVIDIA DGX Spark systems with vLLM tensor parallelism, FlashInfer sparse MLA, FP8 KV cache, and MTP3 speculative decoding.

**Measured production result:** **29.74 tok/s at C1** and **101.74 aggregate tok/s at C8**, with a configured 262,144-token context window.

## Performance

Two matched cold starts. Each scenario used one warm-up wave followed by three measured waves with 512 completion tokens per request. The primary number is the arithmetic mean of the two cold-run medians.

| Concurrency | Aggregate decode throughput | Best cold-run median | Mean TTFT |
|---:|---:|---:|---:|
| C1 | **29.74 tok/s** | 30.11 tok/s | 0.287 s |
| C4 | **68.07 tok/s** | 68.82 tok/s | 0.740 s |
| C6 | **87.39 tok/s** | 87.43 tok/s | 0.776 s |
| C8 | **101.74 tok/s** | 102.34 tok/s | 0.826 s |

C4/C6/C8 are **aggregate throughput**, not per-request speed. Full machine-readable methodology and values are in [`results/production-tp2-mtp3-262k.json`](results/production-tp2-mtp3-262k.json).

### Validated production profile

| Component | Configuration |
|---|---|
| Hardware | 2× NVIDIA DGX Spark / GB10, RoCE interconnect |
| Checkpoint | [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) at `9e0d74e3cef17f634e84fb8e2223707e02616290` |
| Tensor parallelism | TP2, 32 local query heads per rank |
| vLLM | `0.1.dev20051+g487ecf187` |
| FlashInfer | `0.6.17`, rebuilt for `sm_121a` |
| Attention | `FLASHINFER_MLA_SPARSE_SM120` |
| MoE | Marlin NVFP4 |
| KV cache | `fp8_ds_mla`, block size 256 |
| Speculation | MTP3; measured acceptance 68.06% |
| Context | 262,144 configured; 140,012-token prompt completed |
| Scheduler | `max_num_seqs=12`, 8,192 batched tokens |
| Execution | eager |
| Thinking | enabled by default at High; `glm45` reasoning parser |

The production run also passed deterministic arithmetic (`17 × 23 = 391`), tool-call parsing, endpoint health, and a 140K-token prompt. It completed with zero container restarts, zero OOM kills, and 372,827 realized KV-cache tokens.

### Thinking enabled by default

The rank launcher now starts GLM-5.3 with thinking enabled at **High**:

```json
{"enable_thinking":true,"reasoning_effort":"high"}
```

The `glm45` reasoning parser keeps reasoning separate from final content in the OpenAI-compatible response. Clients can still send an explicit `reasoning_effort` when they need a different supported level; requests that omit it inherit High from the server.

## Why this adapter exists

GLM-5.3 is logically NoPE, while the current FlashInfer SM120/SM121 `fp8_ds_mla` GLM_NSA kernel uses a fixed physical ABI:

| Contract | Logical GLM-5.3 | Physical kernel ABI |
|---|---:|---:|
| absorbed query | 512 | 512 + 64 zero padding |
| KV latent | 512 FP8 | 512 FP8 |
| scale metadata | — | four FP32 values |
| positional payload | none | 64 BF16 zeros |
| cache bytes/token | — | 656 |
| architecture sparse top-k | 2048 | — |
| aligned sparse-buffer capacity | — | 2176 |

TP2 presents **32 local query heads** to each rank. The adapter therefore adds exact FlashInfer decode and prefill specializations for both `(32, 2176)` and the TP1 control `(64, 2176)`, pads the physical NoPE ABI with zeros, preserves per-row valid lengths, and rebuilds the AOT module for GB10/SM121a.

No model tensor or Hugging Face configuration field is rewritten. See [`docs/adaptation.md`](docs/adaptation.md) and [`docs/tp2-h32-specialization.md`](docs/tp2-h32-specialization.md).

## Repository contents

- `container/` — digest-pinned ARM64 vLLM image and FlashInfer specialization patch.
- `overlay/` — small runtime compatibility overlay for the GLM NoPE physical ABI.
- `scripts/build-image.sh` — reproducible local image build.
- `scripts/rank-tp2.sh` — parameterized rank launcher for two DGX Sparks.
- `scripts/start-tp2.sh` / `scripts/stop-tp2.sh` — worker-first TP2 lifecycle.
- `bench/benchmark.py` — the frozen C1/C4/C6/C8 benchmark harness.
- `results/` — sanitized production benchmark receipt.

Model weights, Docker layers, CUDA caches, host configuration, credentials, and private logs are intentionally not included.

## Quick start

### 1. Prepare both nodes

Requirements:

- two NVIDIA DGX Spark systems with Docker and NVIDIA Container Toolkit;
- the same repository path and built image on both nodes;
- passwordless SSH from the head to the worker;
- a working RoCE interface/HCA on both nodes;
- enough local storage for the approximately 181 GiB checkpoint.

Acquire the checkpoint independently at the pinned revision:

```bash
hf download LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  --revision 9e0d74e3cef17f634e84fb8e2223707e02616290 \
  --local-dir /models/GLM-5.3-Flash-NVFP4
```

### 2. Build the runtime image on both nodes

```bash
git clone https://github.com/LimeChain/glm-5.3-flash-2xdgx-spark-vllm.git
cd glm-5.3-flash-2xdgx-spark-vllm
IMAGE=glm53-sm121:local ./scripts/build-image.sh
```

The build starts from the immutable ARM64 base image in [`config/versions.env`](config/versions.env), patches the exact FlashInfer sources, recompiles the SM121a AOT module, and validates the resulting runtime contract.

### 3. Configure the cluster

```bash
cp config/cluster.env.example config/cluster.env
$EDITOR config/cluster.env
```

Set the worker SSH target, repository/model/cache paths, fabric addresses, interface, and HCA names. Copy the completed non-secret cluster config to the same repository path on the worker.

### 4. Preflight and start

Run on the head node:

```bash
./scripts/preflight-tp2.sh
./scripts/start-tp2.sh
```

The API binds to loopback on the head by default. Verify it:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

Stop both ranks with:

```bash
./scripts/stop-tp2.sh
```

Detailed setup and safety notes are in [`docs/two-node-deployment.md`](docs/two-node-deployment.md).

## Reproduce the benchmark

With a drained local endpoint:

```bash
python3 bench/benchmark.py \
  --base http://127.0.0.1:8000 \
  --label my-cold-run-1 \
  --output results/my-cold-run-1.json
```

Run it after each independent cold start. The harness is frozen to C1/C4/C6/C8, one warm-up wave, three measured waves, and 512 completion tokens. Throughput excludes the first streamed token and measures active delivery time.

## Validation boundaries

- **262,144 tokens is the configured context window.** The completed long-context qualification used a 140,012-token prompt.
- `max_num_seqs=12` is an admission ceiling, not a C12 throughput result. During the admission test, nine requests ran and three waited; no C12 throughput is claimed.
- The benchmark does not establish model-quality equivalence, global speed leadership, or a matched comparison with other public recipes.
- C1/C4/C6/C8 results bind the exact production source/profile identified in the receipt. Re-run before publishing numbers for a materially changed image or configuration.
- The published throughput receipt predates this default-thinking launcher change. Re-run it before representing the table as a matched High-reasoning benchmark.
- Eager execution is the qualified profile in this release.

## Provenance and licensing

This snapshot builds on the GLM-5.3 GB10 adapter work by [`cyijun`](https://github.com/cyijun/glm-5.3-flash-nvfp4-gb10), with the TP2 32-local-head specialization and production qualification performed by Christian Veselinov / LimeChain.

The source adapter did not expose a repository-level license when this snapshot was prepared. Existing per-file and upstream notices remain authoritative; this repository does not invent or imply a new license grant. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Credits

- [vLLM](https://github.com/vllm-project/vllm)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [`cyijun/glm-5.3-flash-nvfp4-gb10`](https://github.com/cyijun/glm-5.3-flash-nvfp4-gb10)
- [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
