#!/usr/bin/env python3
"""Explicit experimental workload receipts; does not replace the frozen campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from benchmark import metric_snapshot, request_json, run_wave, utc_now

PROMPTS = {
    "count": "Generate a continuous deterministic sequence of decimal integers starting at 1, separated only by commas and spaces. Continue until the output limit; do not explain, summarize, or stop early.",
    "code": "Write a complete Python implementation of a thread-safe bounded LRU cache with expiration using only the standard library. Include type hints, clear docstrings, and a runnable unittest suite covering eviction, expiration, capacity, and concurrent access. Output the code directly.",
    "reasoning": "A delivery service has 3 vans of capacity 7, 9, and 12 boxes. It must move 83 boxes. Every round can use each van once, but each round has a loading cost of 5 minutes and each van trip costs 11, 13, and 17 minutes respectively. Trips in a round run concurrently. Find a schedule minimizing total elapsed time, and explain why it is optimal.",
    "mixed": "Per-client heterogeneous tasks defined in MIXED_PROMPTS.",
}
MIXED_PROMPTS = [
    PROMPTS["code"],
    "Write PostgreSQL queries for monthly revenue growth, rolling three-month averages, and the top three products per region. Explain the window functions and include schema definitions.",
    "Implement a bounded producer-consumer queue in Rust using the standard library. Include graceful shutdown, error handling, and tests.",
    "Write TypeScript debounce and throttle utilities with cancellation and type-safe callbacks, then write tests using a controllable clock.",
    "Prove that the square root of 2 is irrational, then explain how the argument generalizes and where that generalization fails.",
    "Implement Dijkstra's shortest-path algorithm in Python with a heap, reconstruct paths, and include tests for disconnected nodes and zero-weight edges.",
    "Design a PostgreSQL migration that adds a unique customer email constraint to a large live table. Explain duplicate cleanup and rollback.",
    "Implement a CSV parser in JavaScript supporting quoted fields, escaped quotes, embedded newlines, and empty values. Include edge-case tests.",
    "Explain how TCP congestion control changes its sending window after packet loss, and compare slow start, congestion avoidance, and fast recovery using examples.",
    "Write a Go HTTP middleware that implements per-client token-bucket rate limits with bounded memory and race-free cleanup. Include tests.",
    "Design an experiment comparing two website checkout flows. Explain randomization, sample size assumptions, guardrail metrics, and stopping criteria.",
    "Implement a Python interval-merging function for half-open integer intervals. Define how touching and empty intervals behave and include property-based test ideas.",
]


def percentile(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)] if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--workloads", nargs="+", choices=list(PROMPTS), default=["count", "code"])
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--closed-thinking-prefix", action="store_true", help="Explicit assistant prefill for templates that ignore the thinking flag")
    parser.add_argument("--prompt-repeat", type=int, default=0, help="Add deterministic document lines to test prefill")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite receipt")
    if args.tokens < 1 or args.waves < 1 or any(c < 1 for c in args.concurrencies):
        raise SystemExit("positive tokens, waves and concurrency required")
    models = request_json(args.base + "/v1/models")["data"]
    assert len(models) == 1
    opening = metric_snapshot(args.base)
    assert opening["running"] == opening["waiting"] == 0, "endpoint must be drained"
    manifest = json.loads(args.manifest.read_text())
    receipt = {"schema": "glm53-tuning.v1", "label": args.label, "started": utc_now(), "model": models[0], "manifest": manifest, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "scenarios": {}, "pass": True}
    receipt["harness_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("tune.py", "benchmark.py")}
    receipt["cache_config"] = opening.get("cache_config")
    document = "".join(f"Record {i}: warehouse shipment reference alpha; verify quantities and delivery dates.\n" for i in range(args.prompt_repeat))
    for workload in args.workloads:
        for concurrency in args.concurrencies:
            waves = []
            for wave_id in range(args.waves + 1):
                payloads = []
                for client_id in range(concurrency):
                    # Distinct prefixes prevent accidental all-client prefix reuse.
                    task = MIXED_PROMPTS[client_id % len(MIXED_PROMPTS)] if workload == "mixed" else PROMPTS[workload]
                    prompt = f"Request identifier {workload}-{concurrency}-{wave_id}-{client_id}.\n" + document + task
                    payloads.append({"model": models[0]["id"], "messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 17, "max_tokens": args.tokens, "min_tokens": args.tokens, "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": args.thinking, "reasoning_effort": "high"}})
                    if args.closed_thinking_prefix and not args.thinking:
                        payloads[-1]["messages"].append({"role": "assistant", "content": "<think></think>"})
                        payloads[-1].update({"add_generation_prompt": False, "continue_final_message": True})
                wave = run_wave(args.base, payloads[0], concurrency, "warmup" if wave_id == 0 else f"measured-{wave_id}", poll_interval=0.25, per_request_payloads=payloads)
                wave["payload_sha256"] = hashlib.sha256(json.dumps(payloads, sort_keys=True).encode()).hexdigest()
                waves.append(wave)
            measured = waves[1:]
            rows = [r for w in measured for r in w["rows"]]
            valid = all(r["http_status"] == 200 and r["error"] is None and r["stream_done"] and r["completion_tokens"] == args.tokens and r["finish_reason"] == "length" and r["decode_tokens_per_second"] is not None and (args.thinking or r["content_chars"] > 0) for r in rows)
            active = [w["summary"]["aggregate_active_decode_tokens_per_second"] for w in measured]
            total = [sum(r["completion_tokens"] or 0 for r in w["rows"]) / w["summary"]["batch_wall_seconds"] for w in measured]
            latencies = [r["ttft_seconds"] for r in rows if r["ttft_seconds"] is not None]
            per_stream = [r["decode_tokens_per_second"] for r in rows if r["decode_tokens_per_second"] is not None]
            result = {"pass": valid, "requests": len(rows), "median_active_decode_tps": statistics.median(active) if all(v is not None for v in active) else None, "median_end_to_end_tps": statistics.median(total), "median_per_stream_tps": statistics.median(per_stream) if per_stream else None, "median_ttft_s": statistics.median(latencies) if latencies else None, "p95_ttft_s": percentile(latencies, .95), "p95_total_s": percentile([r["total_seconds"] for r in rows], .95), "preemptions": sum(w["summary"]["metric_delta"]["preemptions"] for w in measured), "waves": waves}
            key = f"{workload}-c{concurrency}"
            receipt["scenarios"][key] = result
            receipt["pass"] &= valid
            receipt["finished"] = utc_now()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({"scenario": key, **{k: v for k, v in result.items() if k != "waves"}}), flush=True)
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
