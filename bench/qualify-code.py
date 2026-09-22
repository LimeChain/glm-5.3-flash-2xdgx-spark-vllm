#!/usr/bin/env python3
"""Check a complete coding answer in an isolated, CPU-only Docker container."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

from benchmark import request_json

PROMPT = """Implement the Python function merge_intervals(intervals). Input is a list
of (start, end) integer tuples representing half-open intervals. Return a sorted
list of tuples with overlapping OR touching intervals merged. Discard empty
intervals. Raise ValueError if any start is greater than its end. Do not mutate
the input. Output only the function's Python code, with no imports or tests."""
CASES = [([], []), ([(1, 1)], []), ([(1, 3), (3, 5)], [(1, 5)]),
         ([(6, 8), (1, 4), (2, 3)], [(1, 4), (6, 8)]),
         ([(-4, -1), (-2, 2)], [(-4, 2)]),
         ([(1, 2), (1, 2)], [(1, 2)]),
         ([(0, 10), (3, 4), (5, 5)], [(0, 10)])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--image", required=True, help="Existing local runtime image containing Python 3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite receipt")
    model = request_json(args.base + "/v1/models")["data"][0]["id"]
    receipt = {"model": model, "pass": True, "results": []}
    for thinking in (False, True):
        started = time.monotonic()
        row = {"thinking": thinking}
        try:
            response = request_json(args.base + "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": PROMPT}], "temperature": 0, "seed": 17, "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": thinking, "reasoning_effort": "high"}}, timeout=600)
            row.update({"answer_seconds": time.monotonic() - started, "response": response})
            choice = response["choices"][0]
            source = (choice["message"].get("content") or "").strip()
            fenced = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", source, re.DOTALL)
            if fenced:
                source = fenced.group(1)
            compile(source, "model_answer.py", "exec")
            harness = f'''import copy
namespace = {{}}
exec(compile({source!r}, "model_answer.py", "exec"), namespace)
fn = namespace["merge_intervals"]
for inputs, expected in {CASES!r}:
    original = copy.deepcopy(inputs)
    assert fn(inputs) == expected, (inputs, expected)
    assert inputs == original, "input was mutated"
try:
    fn([(2, 1)])
except ValueError:
    pass
else:
    raise AssertionError("invalid interval was not rejected")
print("CODE_QUALIFICATION_OK")
'''
            container = "glm-code-check-" + uuid.uuid4().hex
            command = ["docker", "run", "--rm", "--name", container, "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "32", "--memory", "128m", "--cpus", "1", "--ulimit", "cpu=3:3", "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "-e", "GLM53_GB10_PATCH_DISABLE=1", "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python3", "-i", args.image, "-"]
            try:
                result = subprocess.run(command, input=harness, capture_output=True, text=True, timeout=15)
            finally:
                subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            row.update({"pass": choice["finish_reason"] == "stop" and result.returncode == 0 and result.stdout.strip().endswith("CODE_QUALIFICATION_OK"), "test_exit": result.returncode, "test_stdout": result.stdout, "test_stderr": result.stderr})
        except Exception as exc:
            row.update({"pass": False, "error": f"{type(exc).__name__}: {exc}"})
        receipt["results"].append(row)
        receipt["pass"] &= row["pass"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "response"}), flush=True)
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
