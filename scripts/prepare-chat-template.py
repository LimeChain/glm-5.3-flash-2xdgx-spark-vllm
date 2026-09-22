#!/usr/bin/env python3
"""Derive an explicit thinking-mode template without changing the checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = "41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3"
ORIGINAL = "<|assistant|>{{- '<think>' -}}"
REPLACEMENT = "<|assistant|>{{- '<think></think>' if enable_thinking is defined and not enable_thinking else '<think>' -}}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise SystemExit("keep the verified checkpoint immutable; choose a separate output")
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise SystemExit("unexpected source template: inspect the new revision before adapting it")
    template = source.decode()
    assert template.count(ORIGINAL) == 1
    result = template.replace(ORIGINAL, REPLACEMENT).encode()
    if args.output.exists() and args.output.read_bytes() != result:
        raise SystemExit("refusing to replace a different template")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)
    print(json.dumps({"source_sha256": SOURCE_SHA256, "output_sha256": hashlib.sha256(result).hexdigest(), "output": str(args.output)}))


if __name__ == "__main__":
    main()
