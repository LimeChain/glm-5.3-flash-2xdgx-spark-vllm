# Measured two-Spark tuning

Preserve the qualified profile as a regression control. Set `PROFILE=experimental`
in a separate cluster configuration to vary speculation, scheduler limits,
context, memory utilization, execution mode, or reasoning parser. Experimental
settings carry no performance or quality qualification until measured on the
target machines.

## Deployment gates

Preflight checks both ranks without starting either, verifies all indexed model
shards exist, and compares image identity, runtime profile, and model metadata.
For checkpoint byte verification, generate a manifest from the pinned Hugging
Face revision with entries `name`, `size`, and `sha256` (the LFS hash, or null for
non-LFS files), then run `scripts/verify-checkpoint.py` on both machines. Compare
the resulting file lists; metadata presence alone is not byte parity.

Build the image once and distribute the same artifact. Modified tracked build
inputs are distinguished with a diff hash in the OCI revision label. This label
is not a replacement for archiving the exact source, untracked build inputs,
image ID, and effective deployment configuration.

Startup captures diagnostic logs and stops newly attempted ranks if readiness
fails. Preserve stopped containers and logs until the cause is understood.

## Network before model loading

Map Ethernet interfaces to RoCE HCAs with `ibdev2netdev`. Confirm link state,
addressing, and matching RoCE-v2 GID indices. When selecting multiple HCAs, the
configured GID index must be appropriate on every selected HCA.

`bench/fabric.py` measures two-rank BF16 all-reduce correctness and latency in
the serving image. Start `scripts/bench-fabric.sh` on rank 1 and rank 0 with
`HEAD_IP`, `FABRIC_IF`, `GID_INDEX`, and `IMAGE` set; pass the rank, HCA list, and
cross-NIC setting as arguments. Both ranks must receive the same protocol and
channel configuration. Optional `BENCH_GRAPHS=1` also measures captured
collectives; graph objects are released before destroying the process group.

Compare one versus both validated interfaces, then optional `NCCL_CHANNELS` and
`NCCL_PROTO_VALUE` settings. Include small decode-size transfers and larger
prefill-size transfers. Stop unrelated fabric transfers during measurements.
A microbenchmark improvement is not an end-to-end model speedup.

## Correctness before throughput

Run `tests/gpu_attention_reference.py` inside the built image on each GPU. It
compares nonzero packed FP8 cache values with unequal per-block scales against
a float32 attention reference for H32/H64 and both sides of the decode/prefill
dispatch boundary. The test covers partial sparse lengths and tail capacity.
It is a targeted numerical regression, not a model-quality equivalence claim.

After serving starts, run `bench/qualify.py` against the private endpoint. It
checks arithmetic with thinking both enabled and disabled, plus an automatic
tool call. Inspect the saved responses, and expand task-level checks for the
intended production workload.

`bench/qualify-code.py` requests a complete interval-merging function in both
thinking modes, measures answer completion time, and checks edge cases and input
immutability. It runs generated code in a temporary CPU-only Docker container
with no network or host mounts, read-only storage, and resource/time limits.
Pass `--image` with an existing local image containing Python 3. This is one
coding smoke test, not a coding benchmark suite.

```bash
python3 bench/qualify.py --base http://127.0.0.1:8000 \
  --output results/quality.json
```

### Thinking-mode template correction

The pinned checkpoint's template always ends the new assistant prompt with
`<think>` and ignores `enable_thinking=false`. The pinned `glm45` parser does
honor that flag. In live tests this merged reasoning and final text into answers
such as `17 × 23 = 391391`. Merely passing the flag does not disable reasoning.

Generate an adapted template separately from the verified checkpoint:

```bash
python3 scripts/prepare-chat-template.py \
  --source /models/GLM-5.3-Flash-NVFP4/chat_template.jinja \
  --output /deployment/chat-template.jinja
```

Set `PROFILE=experimental` and `CHAT_TEMPLATE_HOST` to that output on each
node. The launcher mounts it read-only and compares its hash across ranks.
The adapter verifies the exact source hash and closes the initial thinking tag
only when `enable_thinking=false`; default/High prompts and existing message
history stay unchanged. Run `tests/template_reference.py` with Jinja2 installed
to check both templates, then rerun the live quality gate after restart.

For an already-loaded server using the original template, `bench/qualify.py`
and `bench/tune.py` support `--closed-thinking-prefix`, which explicitly prefills
an empty assistant reasoning block. Record that flag in comparisons; do not
silently relabel old flag-only receipts as non-thinking results.

`bench/check-context.py` checks retrieval at 8K and approximately 60K prompt
tokens, plus a complete streamed High-reasoning answer. Its default sizes fit
the 64K experimental profile. Run it on a drained endpoint and retain actual
prompt usage and responses. These are smoke checks, not a broad quality eval.
`bench/telemetry.py` records GPU activity, clocks, temperature, available host
memory, and cumulative swap counters on each node; compare counter deltas over
the measured interval.

## Throughput campaign

The historical `bench/benchmark.py` remains the frozen counting workload with
the `enable_thinking=false` request flag. On the unmodified pinned template,
that flag does not actually disable model reasoning. It now rejects interrupted
streams and requires the SSE completion marker. Its receipts do not qualify
either correct thinking-mode control or High-reasoning output quality.

`bench/tune.py` adds distinct per-client counting, coding, reasoning, and mixed
prompts. The mixed workload assigns different tasks to concurrent clients to
exercise different expert-routing paths. Supply an inventory JSON with
`--manifest`; it is copied into each
receipt together with request settings and payload hashes. The inventory should
include both image identities, checkpoint revision/hash manifest, source tree
identity, exact engine arguments, fabric settings, and hardware/software
versions. Use separate output files and consistent workloads across candidates.

Example, from the head with a drained endpoint:

```bash
python3 bench/tune.py --base http://127.0.0.1:8000 \
  --label candidate-a --manifest deployment.json \
  --workloads count code --concurrencies 1 4 8 12 \
  --tokens 512 --waves 3 --output results/candidate-a.json
```

Use `--thinking` for a separate reasoning-enabled run, and `--prompt-repeat`
to add deterministic document records. Receipts retain actual API prompt-token
usage and time to first final-content chunk. Short fixed-output reasoning tests
measure generation performance, not necessarily completion of a useful answer.

Track median active-decode and end-to-end throughput, per-stream speed, TTFT,
completion latency, waiting, preemptions, memory, and thermal behavior. Tail
statistics from a small screening run are provisional; use a larger workload
and soak test for the final candidate. Re-run finalists after independent
restarts and reject speedups accompanied by corrupted or missing outputs.

## Useful comparison order

1. Eager baseline with a validated fabric.
2. MTP depth 0/1/2/3 at otherwise matching settings; see the
   [controlled comparison protocol](mtp-comparison.md) for complete-answer checks
   and selection criteria.
3. Supported CUDA-graph modes, checking memory and hybrid-state compatibility.
4. Batched-token budgets and concurrency admission ceilings.
5. Longer prompts and mixed traffic, followed by repeated finalist runs.

Maximum aggregate throughput and minimum single-request latency can select
different profiles. Publish both results with their workload and reasoning
mode instead of describing one configuration as universally fastest.
