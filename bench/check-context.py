#!/usr/bin/env python3
"""Retrieve known values from long prompts and verify streamed reasoning output."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

from benchmark import request_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt-tokens", nargs="+", type=int, default=[8192, 60000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite receipt")
    model = request_json(args.base + "/v1/models")["data"][0]["id"]
    receipt = {"model": model, "pass": True, "results": []}

    def save(result):
        receipt["results"].append(result)
        receipt["pass"] &= result["pass"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("response", "reasoning", "content")}), flush=True)

    for target in args.prompt_tokens:
        if not 1024 <= target <= 64000:
            raise SystemExit("prompt target must be between 1024 and 64000 tokens")
        count = max(1, target // 20)
        for _ in range(8):
            records = [f"Record {i}: ordinary shipment; destination warehouse; status completed; checksum {(i * 7919) % 1000000:06d}." for i in range(count)]
            records[count // 10] = "SPECIAL RECORD ALPHA: authorization code PINE-4729."
            records[count * 7 // 10] = "SPECIAL RECORD BETA: authorization code LIME-8136."
            prompt = "Read these records and recover the two special authorization codes.\n" + "\n".join(records) + "\nReturn only the ALPHA code followed by the BETA code, separated by one space."
            tokens = request_json(args.base + "/tokenize", {"model": model, "prompt": prompt})["count"]
            if target * .97 <= tokens <= target:
                break
            count = max(2, int(count * (target - 100) / tokens))
        started = time.monotonic()
        try:
            response = request_json(args.base + "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}}, timeout=1800)
            content = response["choices"][0]["message"].get("content") or ""
            save({"name": f"retrieval-{target}", "pass": content.strip() == "PINE-4729 LIME-8136", "elapsed_s": time.monotonic() - started, "raw_prompt_tokens": tokens, "response": response})
        except Exception as exc:
            save({"name": f"retrieval-{target}", "pass": False, "elapsed_s": time.monotonic() - started, "error": str(exc)})

    payload = {"model": model, "messages": [{"role": "user", "content": "What is 17 multiplied by 23? Return only the integer in your final answer."}], "temperature": 0, "max_tokens": 2048, "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}
    result = {"name": "streamed-high-reasoning", "content": "", "reasoning": "", "done": False, "finish_reason": None}
    started = time.monotonic()
    try:
        req = urllib.request.Request(args.base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            for raw in response:
                line = raw.decode().strip()
                if line == "data: [DONE]":
                    result["done"] = True
                    break
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    result["usage"] = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    result["content"] += delta.get("content") or ""
                    result["reasoning"] += delta.get("reasoning") or delta.get("reasoning_content") or ""
                    result["finish_reason"] = choice.get("finish_reason") or result["finish_reason"]
        result["pass"] = result["done"] and result["finish_reason"] == "stop" and result["content"].strip().rstrip(".") == "391" and bool(result["reasoning"])
    except Exception as exc:
        result.update({"pass": False, "error": str(exc)})
    result["elapsed_s"] = time.monotonic() - started
    save(result)
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
