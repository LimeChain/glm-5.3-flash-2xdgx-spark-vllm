#!/usr/bin/env python3
"""Render the original and derived chat templates with Jinja2 (no GPU needed)."""
import argparse
import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--original", type=Path, required=True)
parser.add_argument("--adapted", type=Path, required=True)
args = parser.parse_args()
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, **kwargs)
original = env.from_string(args.original.read_text())
adapted = env.from_string(args.adapted.read_text())
tool = {"type": "function", "function": {"name": "multiply", "description": "Multiply integers", "parameters": {"type": "object"}}}
scenarios = [
    {"messages": [{"role": "user", "content": "Return 391."}]},
    {"messages": [{"role": "system", "content": "Be concise."}, {"role": "user", "content": [{"type": "text", "text": "Return 391."}]}]},
    {"messages": [{"role": "user", "content": "Multiply 17 and 23."}], "tools": [tool]},
    {"messages": [{"role": "user", "content": "Multiply 17 and 23."}, {"role": "assistant", "content": "", "reasoning_content": "Use multiplication.", "tool_calls": [{"id": "call-1", "function": {"name": "multiply", "arguments": {"a": 17, "b": 23}}}]}, {"role": "tool", "tool_call_id": "call-1", "content": "391"}], "tools": [tool]},
]
checks = 0
for scenario in scenarios:
    for effort in ("low", "high", "max"):
        common = {"add_generation_prompt": True, "reasoning_effort": effort, **scenario}
        expected = original.render(**common)
        assert expected.endswith("<|assistant|><think>")
        assert adapted.render(**common) == expected
        assert adapted.render(**common, enable_thinking=True) == expected
        assert adapted.render(**common, enable_thinking=False) == expected + "</think>"
        checks += 3
        # Existing turns/tool history are identical when no new assistant is appended.
        common["add_generation_prompt"] = False
        assert adapted.render(**common, enable_thinking=False) == original.render(**common)
        checks += 1
print(json.dumps({"pass": True, "render_checks": checks}))
