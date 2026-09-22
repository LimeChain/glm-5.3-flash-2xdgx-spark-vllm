# MTP0/1/2/3 comparison receipts

See the [hardware report](../../docs/mtp-results-2026-09-22.md) and
[protocol](../../docs/mtp-comparison.md) for methods, selection, and limitations.

Each `graphs-mtpN-s16-m85-*` group describes one fresh model start:

- `tasks.json`: seven complete-answer cases, each with one warm-up and three
  measured trials. Includes actual usage, timings, synthetic outputs, execution
  checks, payload hashes and harness identity. Failed cases are intentionally
  retained; MTP0/1/3 have `pass=false`.
- `throughput.json`: High-reasoning mixed traffic at C1/C8/C16, exactly 256
  generated tokens per request, one warm-up and three measured waves. Each
  profile contains 100 requests including warm-ups.
- `quality.json`: four arithmetic checks and automatic tool-call selection.
- `context.json`: approximately 8K/32K exact retrieval and streamed reasoning.
- `runtime-*.json`: actual speculative settings and image identity on both ranks.

Additional summaries:

- `comparison.json`: task medians, actual output-token counts, full-suite gates,
  throughput including warm-up validity, and matching input/source hash checks.
  A null short-coding geometric-mean ratio means failed tasks prevent a qualified
  comparison. It is not zero latency or a passing promotion decision.
- `startup-summary.json`: per-rank KV/graph allocation, selected fabric, and
  source-log hashes.
- `telemetry-summary.json`: sampled serving intervals with startup excluded,
  host memory, GPU activity/temperature/clocks, and swap deltas. The head had
  15.76 MiB of new swap-out during MTP3; the campaign was not entirely swap-free.
- `prior-mtp3-allocation.json`: both ranks from the previous campaign, correcting
  interpretation of the previously reported head-only KV-memory figure.
- `final-state.json` and `gateway-final.json`: retained MTP3 service health and
  basic gateway checks after the campaign.

Source receipt/log hashes identify the preserved private originals, **not** the
sanitized JSON files here. Deployment addresses, host names and local paths are
omitted. These files contain synthetic prompts/outputs only. Preliminary harness
pilots are not part of this matched comparison.
