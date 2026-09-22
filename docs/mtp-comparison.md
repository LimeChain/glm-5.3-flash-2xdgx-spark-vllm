# Controlled MTP-depth comparison

The completed campaign and its failed cases are documented in the
[2026-09-22 hardware results](mtp-results-2026-09-22.md).

This experiment compares zero, one, two and three speculative draft tokens on
the measured 64K graph profile. Keep the checkpoint, image, chat template,
network settings, CPU availability, memory fraction, graph configuration,
batch-token budget and maximum sequence count fixed. Realized cache and graph
allocations can still change with MTP depth and must be recorded.

The fixed settings come from the [64K measured profile](tuning-2026-09-21.md#reproduce-the-candidate),
including graph capture sizes `[4,8,16,24,32,48,64]`. The experiment tests
changing MTP depth in that profile. It does not optimize the capture grid or
other engine settings independently for each depth, and cannot establish the
best achievable performance of every depth.

Use a fresh model start for each candidate, verify the effective speculative
configuration on both ranks, and retain every failure. The initial campaign
order is MTP2, MTP1, MTP0, then MTP3, ending on the existing control. Preliminary
harness smoke checks on the already-running MTP3 service are separate from the
fresh-start comparison. This order is not randomized and one start per profile
does not establish cold-start variance.
The model processes restart, while the existing per-node compiled-kernel and
autotune caches are retained. This is not a comparison from empty compilation
caches.

## Complete-answer measurements

`bench/task_latency.py` streams complete answers and retains actual token usage,
first-token and first-final-content latency, answer completion time, output,
payload hash, harness hash, and the supplied deployment manifest. It measures:

- Interval merging and lexical topological sorting, each with thinking disabled
  and with High reasoning enabled. Generated Python is checked for specified
  semantics and input immutability in a temporary CPU-only container without
  network or host mounts, with read-only storage and resource/time limits.
- Interval-merging answers following approximately 8K and 32K tokens of synthetic
  background records, with High reasoning. A distinct identifier before the
  document avoids full-document prefix-cache reuse. These are synthetic prefill
  tests, not coding tasks against a real repository.
- An automatic tool request, checked arguments, a local synthetic tool result,
  and a correct final answer. Round-trip timing includes both API requests and
  local orchestration; coding answer timing excludes the subsequent test execution.

Each case has one warm-up and three measured requests, temperature zero and
seed 17. Generation may stop normally before the 4,096-token ceiling; a length
stop, missing usage, missing SSE completion marker, stream error or failed semantic
check disqualifies the case. Partial receipts keep `complete=false` and
`pass=false`. Complete-answer latency can change because the model emits different
amounts of text, so retain usage and compare fixed-length generation separately.

```bash
python3 bench/task_latency.py --base http://127.0.0.1:8000 \
  --image your-verified-image --manifest deployment.json \
  --waves 3 --context-sizes 8192 32768 --output results/mtp2-tasks.json

python3 bench/check-context.py --base http://127.0.0.1:8000 \
  --prompt-tokens 8192 32768 --output results/mtp2-context.json

python3 bench/tune.py --base http://127.0.0.1:8000 \
  --label mtp2 --manifest deployment.json --workloads mixed \
  --concurrencies 1 8 16 --tokens 256 --waves 3 --thinking \
  --output results/mtp2-throughput.json
```

The fixed-length comparison uses High reasoning, distinct prompts, one warm-up
and three measured waves at C1/C8/C16. Generated-token rates include reasoning.
C1 uses the mixed workload's first coding task. Do not compare these 256-token
rates directly with the earlier 512-token High-reasoning campaign.

## Selection rule

Retain MTP3 unless another fully qualified profile shows a useful, repeatable
benefit. Before reading the candidate results, use these practical screening
thresholds (not statistical confidence bounds):

- All API, task, retrieval and stream checks pass; no OOMs or engine failures.
- At least a 5% reduction in the geometric mean of median answer times for the
  four short coding cases, supported by at least a 5% improvement in fixed-length
  C1 generation throughput. A shorter answer alone is insufficient.
- C8 and C16 aggregate throughput each remain within 5% of the control, with no
  new preemptions. Report any latency/throughput tradeoff rather than declaring a
  universal winner.
- Investigate a greater than 10% regression in either long-prompt case before
  promotion. Inspect steady-serving memory, swap deltas and thermal telemetry.

A prospective new winner receives a further fresh-start confirmation of the
primary latency/throughput measurements before becoming the selected deployment.
Small or conflicting differences leave the existing control in place. The test
set is a targeted smoke/performance suite, not broad model-quality qualification.
