# Measured two-Spark tuning receipts

See [the campaign report](../../docs/tuning-2026-09-21.md) for configuration,
methodology, results, and limits.

- `dual-eager-mtp3-screen`: eager counting/coding control, C1/4/8/12.
- `dual-eager-mtp3-mixed`: heterogeneous eager control, C1/8/12.
- `dual-eager-mtp3-fastcpu-screen`: rejected CPU-affinity experiment.
- `dual-graphs-mtp3-s16-m85-screen` and `-mixed`: combined graph/cache/admission candidate.
- `dual-graphs-mtp3-s16-m85-high-mixed`: separate 512-token High-reasoning measurements.
- `dual-graphs-mtp3-s16-m85-repeat`: five-wave C4/C8 follow-up, same running candidate.
- `dual-eager-mtp3-quality`: preserved **failed** ordinary non-thinking API checks with the original template.
- `dual-eager-mtp3-prefix-quality`: successful explicit empty-thinking-prefill workaround.
- Candidate `-quality`, `-context`, and `-code-quality`: ordinary API and completed-task checks with the corrected template.
- `gpu-boundary-*`: direct packed FP8 attention kernel comparisons with a float32 reference.
- `nccl-*` and `pynccl-*`: collective microbenchmarks; these are not model throughput.

Synthetic per-request timings and wave summaries are retained. Private deployment
paths, addresses and host names have been omitted from benchmark manifests and
CLI arguments; each published file includes the SHA256 of its original private
receipt. A filename is an experiment label, not a statement that every attempted
experiment passed or that a profile is universally fastest. Read `pass`, the
arguments, and the report before comparing rates.
