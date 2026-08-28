#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.request
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL = "glm-5.3-flash-nvfp4-sm121"
PROMPT = (
    "Generate a continuous deterministic sequence of decimal integers starting at 1, "
    "separated only by commas and spaces. Continue until the output limit; do not "
    "explain, summarize, or stop early."
)
METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "preemptions": "vllm:num_preemptions_total",
    "request_errors": "vllm:request_success_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def metric_snapshot(base: str) -> dict[str, Any]:
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        text = response.read().decode("utf-8", "replace")
    out: dict[str, Any] = {key: 0.0 for key in METRICS}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            left, raw_value = line.rsplit(" ", 1)
            value = float(raw_value)
        except ValueError:
            continue
        if left.startswith("vllm:cache_config_info{"):
            labels: dict[str, str] = {}
            for item in left[left.find("{") + 1 : left.rfind("}")].split(","):
                name, _, raw = item.partition("=")
                labels[name] = raw.strip('"')
            out["cache_config"] = labels
        for key, metric in METRICS.items():
            if not left.startswith(metric):
                continue
            if key == "request_errors" and 'finished_reason="error"' not in left:
                continue
            out[key] = float(out[key]) + value
    return out


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in (
            "preemptions",
            "request_errors",
            "prompt_tokens",
            "generation_tokens",
            "draft_tokens",
            "accepted_tokens",
        )
    }


def stream_one(
    url: str,
    payload: dict[str, Any],
    label: str,
    gate: threading.Barrier,
) -> dict[str, Any]:
    gate.wait()
    started = time.perf_counter()
    first: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    fingerprint: str | None = None
    content: list[str] = []
    reasoning_chars = 0
    status: int | None = None
    error: str | None = None
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            status = response.status
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                fingerprint = event.get("system_fingerprint") or fingerprint
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or ""
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if first is None and (text or reasoning):
                        first = time.perf_counter()
                    if text:
                        content.append(text)
                    reasoning_chars += len(reasoning)
                    finish_reason = choices[0].get("finish_reason") or finish_reason
                if event.get("usage"):
                    usage = event["usage"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    output = "".join(content)
    completion_tokens = usage.get("completion_tokens")
    decode_tokens = max(int(completion_tokens or 0) - 1, 0)
    decode_seconds = None if first is None else ended - first
    return {
        "label": label,
        "http_status": status,
        "error": error,
        "started_perf": started,
        "first_perf": first,
        "ended_perf": ended,
        "ttft_seconds": None if first is None else first - started,
        "total_seconds": ended - started,
        "decode_seconds": decode_seconds,
        "completion_tokens": completion_tokens,
        "decode_tokens_per_second": (
            None if not decode_seconds else decode_tokens / decode_seconds
        ),
        "finish_reason": finish_reason,
        "system_fingerprint": fingerprint,
        "reasoning_chars": reasoning_chars,
        "content_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def run_wave(
    base: str,
    payload: dict[str, Any],
    concurrency: int,
    label: str,
) -> dict[str, Any]:
    before = metric_snapshot(base)
    gate = threading.Barrier(concurrency + 1)
    samples: list[dict[str, float]] = []
    url = base + "/v1/chat/completions"
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(stream_one, url, payload, f"{label}-client-{index + 1}", gate)
            for index in range(concurrency)
        ]
        gate.wait()
        while not all(future.done() for future in futures):
            sample = metric_snapshot(base)
            samples.append(
                {
                    "t_perf": time.perf_counter(),
                    "running": float(sample["running"]),
                    "waiting": float(sample["waiting"]),
                }
            )
            time.sleep(0.02)
        rows = [future.result() for future in futures]
    after = metric_snapshot(base)
    valid = [row for row in rows if row["decode_tokens_per_second"] is not None]
    first_times = [float(row["first_perf"]) for row in valid]
    end_times = [float(row["ended_perf"]) for row in valid]
    decode_tokens = sum(max(int(row["completion_tokens"] or 0) - 1, 0) for row in valid)
    shared_decode_seconds = (
        max(end_times) - min(first_times) if first_times and end_times else None
    )
    delta = metric_delta(before, after)
    fingerprints = sorted(
        {row["system_fingerprint"] for row in rows if row["system_fingerprint"]}
    )
    public_rows = [
        {key: value for key, value in row.items() if not key.endswith("_perf")}
        for row in rows
    ]
    return {
        "label": label,
        "concurrency": concurrency,
        "rows": public_rows,
        "summary": {
            "http_successes": sum(row["http_status"] == 200 for row in rows),
            "peak_running": max((item["running"] for item in samples), default=0.0),
            "peak_waiting": max((item["waiting"] for item in samples), default=0.0),
            "median_ttft_seconds": (
                statistics.median(float(row["ttft_seconds"]) for row in valid)
                if valid
                else None
            ),
            "median_per_stream_decode_tokens_per_second": (
                statistics.median(
                    float(row["decode_tokens_per_second"]) for row in valid
                )
                if valid
                else None
            ),
            "aggregate_active_decode_tokens_per_second": (
                decode_tokens / shared_decode_seconds if shared_decode_seconds else None
            ),
            "batch_wall_seconds": (
                max(float(row["ended_perf"]) for row in rows)
                - min(float(row["started_perf"]) for row in rows)
            ),
            "metric_delta": delta,
            "system_fingerprints": fingerprints,
        },
    }


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take median of empty values")
    return statistics.median(values)


def summarize_scenario(waves: list[dict[str, Any]], completion_tokens: int) -> dict[str, Any]:
    measured = [wave for wave in waves if wave["label"].startswith("measured-")]
    summaries = [wave["summary"] for wave in measured]
    all_rows = [row for wave in measured for row in wave["rows"]]
    draft = sum(float(item["metric_delta"]["draft_tokens"]) for item in summaries)
    accepted = sum(float(item["metric_delta"]["accepted_tokens"]) for item in summaries)
    return {
        "measured_waves": len(measured),
        "median_aggregate_decode_tokens_per_second": median(
            [float(item["aggregate_active_decode_tokens_per_second"]) for item in summaries]
        ),
        "median_per_stream_decode_tokens_per_second": median(
            [float(item["median_per_stream_decode_tokens_per_second"]) for item in summaries]
        ),
        "median_ttft_seconds": median(
            [float(item["median_ttft_seconds"]) for item in summaries]
        ),
        "peak_running": max(float(item["peak_running"]) for item in summaries),
        "peak_waiting": max(float(item["peak_waiting"]) for item in summaries),
        "acceptance_percent": None if not draft else 100.0 * accepted / draft,
        "all_http_success": all(row["http_status"] == 200 for row in all_rows),
        "all_completion_tokens_exact": all(
            row["completion_tokens"] == completion_tokens for row in all_rows
        ),
        "all_finish_reasons_length": all(
            row["finish_reason"] == "length" for row in all_rows
        ),
        "all_reasoning_empty": all(row["reasoning_chars"] == 0 for row in all_rows),
        "request_errors": sum(
            float(item["metric_delta"]["request_errors"]) for item in summaries
        ),
        "preemptions": sum(
            float(item["metric_delta"]["preemptions"]) for item in summaries
        ),
        "fingerprints": sorted(
            {
                fingerprint
                for item in summaries
                for fingerprint in item["system_fingerprints"]
            }
        ),
        "waves": waves,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 6, 8])
    parser.add_argument("--completion-tokens", type=int, default=512)
    parser.add_argument("--warmup-waves", type=int, default=1)
    parser.add_argument("--measured-waves", type=int, default=3)
    args = parser.parse_args()
    if sorted(set(args.concurrencies)) != sorted(args.concurrencies):
        raise SystemExit("concurrencies must be unique and sorted")
    if any(value not in (1, 4, 6, 8) for value in args.concurrencies):
        raise SystemExit("campaign scenarios are frozen to C1/C4/C6/C8")
    if args.completion_tokens != 512:
        raise SystemExit("campaign completion token count is frozen to 512")
    if args.warmup_waves != 1 or args.measured_waves != 3:
        raise SystemExit("campaign wave counts are frozen to one warmup and three measured")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    base = args.base.rstrip("/")
    parsed_base = urlparse(base)
    if parsed_base.scheme != "http" or parsed_base.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise SystemExit("benchmark endpoint must be loopback HTTP")
    started = utc_now()
    models = request_json(base + "/v1/models")
    model_rows = models.get("data") or []
    if [row.get("id") for row in model_rows] != [MODEL]:
        raise RuntimeError("model identity check failed")
    tokenized = request_json(base + "/tokenize", {"model": MODEL, "prompt": PROMPT})
    opening = metric_snapshot(base)
    if float(opening["running"]) or float(opening["waiting"]):
        raise RuntimeError("endpoint is not drained")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": args.completion_tokens,
        "min_tokens": args.completion_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    scenarios: dict[str, Any] = {}
    for concurrency in args.concurrencies:
        waves = [
            run_wave(base, payload, concurrency, "warmup-1")
        ] + [
            run_wave(base, payload, concurrency, f"measured-{index + 1}")
            for index in range(args.measured_waves)
        ]
        scenarios[f"c{concurrency}"] = summarize_scenario(
            waves, args.completion_tokens
        )

    closing = metric_snapshot(base)
    receipt = {
        "schema": "glm53-sm121-throughput-receipt.v1",
        "label": args.label,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "base": base,
        "model": MODEL,
        "max_model_len_from_api": model_rows[0].get("max_model_len"),
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "prompt_tokens": tokenized.get("count"),
        "request_payload": payload,
        "scenarios": scenarios,
        "cache_config": closing.get("cache_config"),
        "drained": float(closing["running"]) == 0.0
        and float(closing["waiting"]) == 0.0,
    }
    receipt["pass"] = bool(receipt["drained"]) and all(
        scenario["all_http_success"]
        and scenario["all_completion_tokens_exact"]
        and scenario["all_finish_reasons_length"]
        and scenario["all_reasoning_empty"]
        and scenario["request_errors"] == 0.0
        and scenario["preemptions"] == 0.0
        and len(scenario["fingerprints"]) == 1
        for scenario in scenarios.values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    compact = {
        "label": args.label,
        "pass": receipt["pass"],
        "prompt_tokens": receipt["prompt_tokens"],
        "kv_tokens": (receipt.get("cache_config") or {}).get("kv_cache_size_tokens"),
        "scenarios": {
            name: {
                "aggregate": value["median_aggregate_decode_tokens_per_second"],
                "per_stream": value["median_per_stream_decode_tokens_per_second"],
                "ttft": value["median_ttft_seconds"],
                "acceptance_percent": value["acceptance_percent"],
                "peak_running": value["peak_running"],
                "peak_waiting": value["peak_waiting"],
            }
            for name, value in scenarios.items()
        },
    }
    print(json.dumps(compact, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
