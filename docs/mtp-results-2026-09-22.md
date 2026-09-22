# MTP-depth hardware comparison — 2026-09-22

**No MTP-depth change was selected.** The existing MTP3 service is retained for
single-request speed with **High reasoning by default**. It reached **26.79 tokens/s
at C1**, versus 21.19 for MTP2. MTP2 had higher concurrent throughput and passed
all 28 complete-answer trials; MTP3 passed 25/28, with failures in non-thinking
topological sorting. MTP3 is not newly qualified as an all-mode correctness winner.

This follows the [64K throughput campaign](tuning-2026-09-21.md). It is a
controlled comparison of MTP0/1/2/3 within that recipe, using complete coding
answers and tool calls as well as fixed-length generation. The
[protocol and promotion criteria](mtp-comparison.md) were committed before the
first candidate completed. Sanitized per-request evidence is in
[`results/2026-09-22`](../results/2026-09-22/).

## Configuration and measurement

Two GB10 nodes, TP2, 65,536 configured context, 16 maximum sequences, 8,192
batched tokens, memory utilization 0.85, Marlin NVFP4, FP8 KV, corrected chat
template, High reasoning by default, and both verified RoCE links. Graph mode
is `FULL_DECODE_ONLY`, compilation mode 0, with captures
`[4,8,16,24,32,48,64]`. There are no CPU-affinity or clock/power overrides.
The only requested engine change is MTP depth; MTP0 omits speculative decoding.
Realized cache allocations differ and are reported below.

The checkpoint remains `9e0d74e3cef17f634e84fb8e2223707e02616290`; both ranks use
image `sha256:36eefbfd9f1470c09c3cff70de5cbfbd5d40b6fdda0f018e7298cdeb5cafffa1`.
Its source label remains
`adbf224a6ab47e28c8fd6112efc8e8ee80d9bb14-dirty-6333f64d10a8`.
The benchmark changes did not rebuild the runtime or modify the model weights.

Every profile receives a fresh model process, in order **2 → 1 → 0 → 3**.
Compiled-kernel/autotune caches persist between starts. Each case has one warm-up
and three measured trials/waves, temperature zero and seed 17. The order was
not randomized; there is one fresh start per profile in this matched comparison.
Small differences and tail percentiles are provisional.

All four profiles have matching benchmark-source and initial input-payload hashes.
The second tool request contains the model's generated messages and tool-call ID,
so that follow-up payload is expected to differ. Effective speculative settings
and the common image identity were checked on both ranks after each start.

## Fixed-length generation

Median active-decode **generated tokens/s**, High reasoning, exactly 256 output
tokens per request. C1 uses the mixed workload's first coding prompt; C8/C16
cycle through heterogeneous tasks. Concurrent rates are aggregate, not the speed
of one user. Reasoning tokens are included. These fixed-length outputs can end
before the task is solved; correctness is checked separately below.

| MTP depth | C1 | C8 | C16 |
|---|---|---|---|
| 0 | 11.86 | 63.78 | 89.06 |
| 1 | 19.78 | 78.12 | 111.17 |
| 2 | 21.19 | 82.72 | 116.89 |
| 3 | 26.79 | 76.80 | 112.64 |

End-to-end throughput includes prompt processing and the entire batch wall time:

| MTP depth | C1 | C8 | C16 |
|---|---|---|---|
| 0 | 11.74 | 63.50 | 88.86 |
| 1 | 19.33 | 77.57 | 110.76 |
| 2 | 20.76 | 77.47 | 116.32 |
| 3 | 26.04 | 76.21 | 111.05 |

First-token latency, in seconds:

| MTP depth | C1 median TTFT | C8 p95 TTFT | C16 p95 TTFT |
|---|---|---|---|
| 0 | 0.295 | 0.940 | 5.659 |
| 1 | 0.324 | 0.857 | 1.193 |
| 2 | 0.299 | 5.176 | 5.714 |
| 3 | 0.310 | 0.893 | 6.051 |

Multi-second outliers remain in some measured waves and are retained in the
reported tails. Their cause was not isolated. Do not compare these 256-token
rates directly with the earlier 512-token High-reasoning results.

## Complete answers

Median seconds from request start to complete answer, excluding the subsequent
CPU execution check. The tool round trip includes both API requests and local
tool orchestration. All long-prompt coding cases use High reasoning.

| Case | MTP0 | MTP1 | MTP2 | MTP3 |
|---|---|---|---|---|
| Interval, thinking off | 10.67 | 7.26 | 6.39 | 4.78 |
| Interval, High | 28.33 | 10.32 | 7.67 | 6.29 |
| Topological sort, thinking off | 26.41† | 24.12† | 10.95 | 7.36† |
| Topological sort, High | 64.15 | 39.67 | 30.18 | 33.58 |
| 8K background + interval | 19.74 | 12.18 | 11.24 | 10.82 |
| 32K background + interval | 33.73 | 31.99 | 26.12 | 24.30 |
| Tool round trip, High | 2.61 | 1.93 | 1.85 | 1.56 |

† Failed case: the median uses all three measured trials, including failed
answers. It is not a median of successful tasks alone. The checker accepts bare
Python or one outer code fence and executes the complete answer against the supplied semantic tests. MTP0 had
one failed measured non-thinking topological-sort answer; MTP1 had two. Each
contained an initial function, explanatory self-correction, and a second code
block. The full answer was not directly executable. MTP3 had two such formatting
failures (warm-up and measured trial 3), plus a **functional failure** in measured
trial 2: it deduplicated adjacency edges but counted duplicate edges in the
in-degrees, incorrectly treating a valid DAG as cyclic. These are real failed
test cases. This small sample does not establish a broad quality ranking or
isolate why outputs differ between profiles.

Including warm-ups, task trials passed **27/28, 26/28, 28/28 and 25/28** for
MTP0/1/2/3 respectively: **106/112 overall**. Every High-reasoning trial passed
(**80/80** across the four profiles), including all tool round trips. All 20
initial API checks, 12 context/stream checks, and **400 fixed-length throughput
requests** passed, including their warm-ups. A successful HTTP stream is not
counted as a successful coding answer unless its execution checks also pass.

Answers differ in length despite matching prompts and sampling settings. Actual
completion-token counts and full synthetic outputs are retained in the receipts;
the fixed-length table is needed to distinguish engine throughput from shorter
answers. The long coding prompts contain **8,199 and 32,074 actual API input
tokens**. They use synthetic background records, not a real repository, with
distinct identifiers before the document to avoid full-document prefix reuse.
Separate exact-value retrieval checks use approximately 8K/32K prompts.

## Allocation and hardware observations

Startup-reported available KV memory is per rank. The graph/cache allocation
varies with MTP depth and startup profiling, even at the same requested memory
fraction. Virtual KV token capacity alone is not an admission guarantee for
this hybrid model.

| MTP depth | Available KV GiB (head / worker) | Graph GiB (head / worker) | Peak C16 KV occupancy |
|---|---|---|---|
| 0 | 8.89 / 8.47 | 0.10 / 0.10 | 26.7% |
| 1 | 7.45 / 5.72 | 0.18 / 0.19 | 62.1% |
| 2 | 6.50 / 6.64 | 0.22 / 0.25 | 75.2% |
| 3 | 6.34 / 6.49 | 0.22 / 0.25 | 98.2% |

All profiles admitted **16 short requests simultaneously**, with no waiting or
preemptions in the measured throughput waves. Both RoCE HCAs were selected by
both ranks. No engine/CUDA error lines were found in the preserved runtime logs.

The previous campaign's 7.43 GiB figure described the head only; its worker had
6.28 GiB. This run's MTP3 cache has **229 blocks / 428,792 virtual tokens**, versus
227 / 425,047 previously. The smaller head-only allocation therefore did not
reduce effective admission. The earlier report now gives both per-rank figures;
no explicit KV-memory override was needed.

Across the serving intervals, sampled median GPU busy time was **95–96%** on
both nodes; this is not percent of peak FLOPS. Maximum sampled temperatures
were **84°C**. Minimum available host memory ranged from **5.62–8.08 GiB** on
the head and **10.00–10.72 GiB** on the worker. Model loading is excluded from
these intervals. MTP3 recorded **4,034 pages (15.76 MiB)** of new swap-out on
the head; the other profile/node intervals recorded none. Some existing pages
were read back from swap. These measurements do not establish a swap-free run
or continuous absence of thermal throttling. Full sampled summaries and gaps
are retained in `telemetry-summary.json`.

The fixed graph capture grid was not tuned independently for each depth.
Higher concurrency ceilings, different token budgets, other MTP depths, mixed
long/short request arrivals, and kernel changes are outside this comparison.
This establishes a choice among the tested configurations, not the maximum
possible performance of the hardware. Kernel profiling and investigation of
first-token outliers remain useful next experiments.

## Selection and deployment

MTP2's fixed-length C1 rate is **20.9% below MTP3**, so it does not meet the
predeclared requirement for a 5% C1 gain. Its C8 and C16 rates are **7.7% and
3.8% higher**, respectively: it is a useful alternative for shared throughput,
with a single-request latency tradeoff. Its full-suite pass on one start does
not establish generally superior model quality.

Because the MTP3 control itself failed the non-thinking sorting case, no
qualified four-case short-coding latency improvement is claimed from medians
that include invalid answers. No candidate met the combined promotion criteria.
The existing deployment is retained for the user's single-request speed
priority, with High reasoning enabled; this is not a claim that MTP3 passed
the entire suite. The non-thinking failures remain a documented limitation.
There is no new runtime image, weight change, or MTP-depth promotion.

After the sweep, both serving containers were running with **zero restarts and
no OOM kills**, the expected MTP3 configuration, a healthy API, zero active or
queued requests, and zero recorded preemptions. The private gateway passed
ordinary non-thinking chat, default-High chat with separate reasoning, and a
basic Responses API request. This is a basic compatibility check, not a broad
client-integration qualification. Campaign telemetry processes were stopped;
the model service was left running.

Source receipt/log SHA256 values bind the sanitized measurements to preserved
private originals. Deployment addresses, host names, user paths and local CLI
paths are omitted. Preliminary MTP3 harness pilots are excluded from the matched
tables above.
