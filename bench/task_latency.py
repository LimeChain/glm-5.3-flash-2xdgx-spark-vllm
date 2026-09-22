#!/usr/bin/env python3
"""Measure complete, checked coding/tool answers, including long-prompt prefill."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time
import urllib.request
import uuid

from benchmark import metric_snapshot, request_json, utc_now

INTERVAL_PROMPT = """Implement the Python function merge_intervals(intervals). Input is a list
of (start, end) integer tuples representing half-open intervals. Return a sorted
list of tuples with overlapping OR touching intervals merged. Discard empty
intervals. Raise ValueError if any start is greater than its end. Do not mutate
the input. Output only the function's Python code, with no imports or tests."""
TOPO_PROMPT = """Implement topological_sort(nodes, edges) in Python. nodes is a list of
unique strings and edges is a list of (source, destination) tuples. Return the
lexicographically smallest valid topological ordering, including isolated nodes.
Ignore duplicate edges. Raise ValueError on a cycle or an edge endpoint absent
from nodes. Do not mutate either input. Output only the function's Python code,
with no imports or tests; built-in functions and types are available."""
INTERVAL_TESTS = '''fn = namespace["merge_intervals"]
for inputs, expected in [([], []), ([(1,1)], []), ([(1,3),(3,5)], [(1,5)]),
    ([(6,8),(1,4),(2,3)], [(1,4),(6,8)]), ([(-4,-1),(-2,2)], [(-4,2)]),
    ([(1,2),(1,2)], [(1,2)]), ([(0,10),(3,4),(5,5)], [(0,10)])]:
    original = copy.deepcopy(inputs)
    assert fn(inputs) == expected, (inputs, expected)
    assert inputs == original, "input mutated"
try:
    fn([(2,1)])
except ValueError:
    pass
else:
    raise AssertionError("invalid interval accepted")
'''
TOPO_TESTS = '''fn = namespace["topological_sort"]
for nodes, edges, expected in [([], [], []), (["c","a","b"], [], ["a","b","c"]),
    (["a","b","c","d"], [("a","b"),("a","b"),("b","d")], ["a","b","c","d"]),
    (["a","b","c"], [("c","a")], ["b","c","a"]),
    (["a","b","c"], [("b","a")], ["b","a","c"])]:
    original = copy.deepcopy((nodes, edges))
    assert fn(nodes, edges) == expected, (nodes, edges, expected)
    assert (nodes, edges) == original, "input mutated"
for nodes, edges in [(["a"], [("a","a")]), (["a","b"], [("a","b"),("b","a")]),
    (["a"], [("a","missing")])]:
    try:
        fn(nodes, edges)
    except ValueError:
        pass
    else:
        raise AssertionError("cycle or missing endpoint accepted")
'''
TOOL = {"type": "function", "function": {"name": "multiply", "description": "Multiply two integers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False}}}


def stream_answer(base, payload):
    started = time.perf_counter()
    row = {"started": utc_now(), "content": "", "reasoning": "", "tool_calls": [], "done": False, "finish_reason": None, "usage": None, "ttft_seconds": None, "first_content_seconds": None, "error": None}
    calls = {}
    try:
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({**payload, "stream": True, "stream_options": {"include_usage": True}}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            for raw in response:
                line = raw.decode().strip()
                if line == "data: [DONE]":
                    row["done"] = True
                    break
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    row["usage"] = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    tc = delta.get("tool_calls") or []
                    has_tool_text = any((c.get("function") or {}).get("name") or (c.get("function") or {}).get("arguments") for c in tc)
                    elapsed = time.perf_counter() - started
                    if row["ttft_seconds"] is None and (text or reasoning or has_tool_text):
                        row["ttft_seconds"] = elapsed
                    if row["first_content_seconds"] is None and text:
                        row["first_content_seconds"] = elapsed
                    row["content"] += text
                    row["reasoning"] += reasoning
                    for c in tc:
                        acc = calls.setdefault(c["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if c.get("id"):
                            acc["id"] = c["id"]
                        for key in ("name", "arguments"):
                            acc["function"][key] += (c.get("function") or {}).get(key) or ""
                    row["finish_reason"] = choice.get("finish_reason") or row["finish_reason"]
        row["tool_calls"] = [calls[i] for i in sorted(calls)]
        if not row["done"] or row["finish_reason"] not in ("stop", "tool_calls") or not row["usage"] or not row["usage"].get("completion_tokens") or row["ttft_seconds"] is None:
            raise RuntimeError("incomplete answer, missing usage, or missing stream termination")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["answer_seconds"] = time.perf_counter() - started
    row["payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return row


def check_code(source, tests, image):
    source = source.strip()
    fenced = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", source, re.DOTALL)
    if fenced:
        source = fenced.group(1)
    compile(source, "model_answer.py", "exec")
    harness = f'import copy\nnamespace = {{}}\nexec(compile({source!r}, "model_answer.py", "exec"), namespace)\n' + tests + '\nprint("TASK_CODE_OK")\n'
    name = "glm-task-check-" + uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "32", "--memory", "128m", "--cpus", "1", "--ulimit", "cpu=3:3", "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "-e", "GLM53_GB10_PATCH_DISABLE=1", "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python3", "-i", image, "-"]
    try:
        result = subprocess.run(command, input=harness, text=True, capture_output=True, timeout=15)
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    return {"pass": result.returncode == 0 and result.stdout.strip() == "TASK_CODE_OK", "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def context_prefix(base, model, target):
    count = max(2, target // 22)
    for _ in range(8):
        text = "The following synthetic records are background data; the coding specification follows.\n" + "\n".join(f"Record {i}: warehouse shipment alpha; checksum {(i*7919)%1000000:06d}; destination north; state completed." for i in range(count)) + "\nEnd of background records.\n"
        tokens = request_json(base + "/tokenize", {"model": model, "prompt": text})["count"]
        if .97 * target <= tokens <= target:
            return text
        count = max(2, int(count * (target - 100) / tokens))
    raise RuntimeError("could not construct target prompt size")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--image", required=True)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--waves", type=int, default=3)
    p.add_argument("--context-sizes", nargs="*", type=int, default=[8192, 32768])
    p.add_argument("--only", nargs="*", help="Optional exact case names for harness smoke tests")
    args = p.parse_args()
    if args.output.exists() or args.waves < 1 or any(n < 1024 or n > 60000 for n in args.context_sizes):
        raise SystemExit("new output, positive waves and context targets 1024..60000 required")
    opening = metric_snapshot(args.base)
    if opening["running"] or opening["waiting"]:
        raise SystemExit("endpoint must be drained")
    model = request_json(args.base + "/v1/models")["data"][0]["id"]
    cases = [(f"{name}-{'high' if thinking else 'off'}", prompt, tests, thinking, 0) for name, prompt, tests in (("interval", INTERVAL_PROMPT, INTERVAL_TESTS), ("topological", TOPO_PROMPT, TOPO_TESTS)) for thinking in (False, True)]
    cases += [(f"interval-{size}-high", INTERVAL_PROMPT, INTERVAL_TESTS, True, size) for size in args.context_sizes]
    cases += [("tool-roundtrip-high", None, None, True, 0)]
    if args.only:
        cases = [c for c in cases if c[0] in args.only]
        if {c[0] for c in cases} != set(args.only):
            raise SystemExit("unknown case name")
    receipt = {"schema": "glm53-task-latency.v1", "complete": False, "pass": False, "started": utc_now(), "manifest": json.loads(args.manifest.read_text()), "model": model, "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cache_config": opening.get("cache_config"), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "cases": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        receipt["updated"] = utc_now()
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    for name, prompt, tests, thinking, context_size in cases:
        prefix = context_prefix(args.base, model, context_size) if context_size else ""
        rows = []
        for wave in range(args.waves + 1):
            row = {"wave": "warmup" if wave == 0 else f"measured-{wave}", "pass": False}
            settings = {"model": model, "temperature": 0, "seed": 17, "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": thinking, "reasoning_effort": "high"}}
            # Identifier before long background prevents full-document prefix-cache reuse.
            identifier = f"Request identifier {name}-{wave}.\n"
            try:
                if prompt:
                    result = stream_answer(args.base, {**settings, "messages": [{"role": "user", "content": identifier + prefix + prompt}]})
                    row.update(result)
                    if result["error"]:
                        raise RuntimeError(result["error"])
                    row["code_check"] = check_code(result["content"], tests, args.image)
                    row["pass"] = result["finish_reason"] == "stop" and row["code_check"]["pass"]
                else:
                    started = time.perf_counter()
                    messages = [{"role": "user", "content": identifier + "Use the multiply tool to compute 17 multiplied by 23. After receiving its result, give only the integer as your final answer."}]
                    first = stream_answer(args.base, {**settings, "messages": messages, "tools": [TOOL], "tool_choice": "auto"})
                    row["tool_response"] = first
                    calls = first["tool_calls"]
                    if first["error"] or first["finish_reason"] != "tool_calls" or len(calls) != 1 or calls[0]["function"]["name"] != "multiply" or json.loads(calls[0]["function"]["arguments"]) != {"a":17,"b":23}:
                        raise RuntimeError("incorrect tool selection or arguments")
                    messages += [{"role": "assistant", "content": first["content"] or None, "reasoning_content": first["reasoning"], "tool_calls": calls}, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "391"}]
                    second = stream_answer(args.base, {**settings, "messages": messages, "tools": [TOOL], "tool_choice": "none"})
                    row.update({"final_response": second, "answer_seconds": time.perf_counter() - started, "ttft_seconds": first["ttft_seconds"], "pass": second["error"] is None and second["finish_reason"] == "stop" and second["content"].strip().rstrip('.') == "391"})
            except Exception as exc:
                row.update({"pass": False, "check_error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            receipt["cases"][name] = {"rows": rows, "complete": False}
            save()
        measured = rows[1:]
        passed = all(r["pass"] for r in rows)
        result = {"complete": True, "pass": passed, "median_answer_seconds": statistics.median(r["answer_seconds"] for r in measured) if all("answer_seconds" in r for r in measured) else None, "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in measured) if all(r.get("ttft_seconds") is not None for r in measured) else None, "rows": rows}
        receipt["cases"][name] = result
        save()
        print(json.dumps({"case": name, **{k:v for k,v in result.items() if k != "rows"}}), flush=True)
    receipt["complete"] = True
    receipt["pass"] = all(c["pass"] for c in receipt["cases"].values())
    receipt["finished"] = utc_now()
    save()
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
