#!/usr/bin/env python3
"""Small API correctness gate before throughput qualification."""
import argparse
import json
from pathlib import Path
import re
import time

from benchmark import request_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", default="http://127.0.0.1:8000")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--closed-thinking-prefix", action="store_true", help="Work around a template that ignores enable_thinking=False")
args = parser.parse_args()
model = request_json(args.base + "/v1/models")["data"][0]["id"]
results = []


def call(name, payload, check):
    started = time.monotonic()
    if args.closed_thinking_prefix and payload.get("chat_template_kwargs", {}).get("enable_thinking") is False:
        payload = {**payload, "messages": [*payload["messages"], {"role": "assistant", "content": "<think></think>"}], "add_generation_prompt": False, "continue_final_message": True}
    try:
        response = request_json(args.base + "/v1/chat/completions", {"model": model, "temperature": 0, "seed": 17, "max_tokens": 2048, **payload}, timeout=300)
        passed = bool(check(response))
        result = {"name": name, "pass": passed, "elapsed_s": time.monotonic() - started, "response": response}
    except Exception as exc:
        result = {"name": name, "pass": False, "elapsed_s": time.monotonic() - started, "error": f"{type(exc).__name__}: {exc}"}
    results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"pass": all(r["pass"] for r in results), "closed_thinking_prefix": args.closed_thinking_prefix, "results": results}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "response"}), flush=True)


for thinking in (False, True):
    for a, b in ((17, 23), (29, 31)):
        call(f"multiply-{a}-{b}-thinking-{thinking}", {"messages": [{"role": "user", "content": f"What is {a} multiplied by {b}? Return only the integer in your final answer."}], "chat_template_kwargs": {"enable_thinking": thinking, "reasoning_effort": "high"}}, lambda r, expected=str(a*b): re.fullmatch(r"\s*" + expected + r"[.\s]*", r["choices"][0]["message"].get("content") or "") is not None)


def tool_check(response):
    calls = response["choices"][0]["message"].get("tool_calls") or []
    return len(calls) == 1 and calls[0]["function"]["name"] == "multiply" and json.loads(calls[0]["function"]["arguments"]) == {"a": 17, "b": 23}


call("tool-call", {"messages": [{"role": "user", "content": "Use the multiply tool to multiply 17 by 23."}], "tools": [{"type": "function", "function": {"name": "multiply", "description": "Multiply two integers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False}}}], "tool_choice": "auto", "chat_template_kwargs": {"enable_thinking": False}}, tool_check)
raise SystemExit(0 if all(r["pass"] for r in results) else 1)
