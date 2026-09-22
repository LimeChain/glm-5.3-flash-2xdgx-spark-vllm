import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from task_latency import stream_answer


class TaskStreamTests(unittest.TestCase):
    def run_stream(self, events, done=True):
        data = b"".join(("data: " + json.dumps(e) + "\n\n").encode() for e in events)
        if done:
            data += b"data: [DONE]\n\n"
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data)):
            return stream_answer("http://127.0.0.1:8000", {"model": "test"})

    def test_fragmented_tool_arguments_are_reassembled(self):
        row = self.run_stream([
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "multiply", "arguments": '{"a":17,'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"b":23}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"completion_tokens": 12}},
        ])
        self.assertIsNone(row["error"])
        self.assertEqual(row["tool_calls"][0]["id"], "call-1")
        self.assertEqual(json.loads(row["tool_calls"][0]["function"]["arguments"]), {"a":17,"b":23})
        self.assertIsNotNone(row["ttft_seconds"])

    def test_finished_text_without_done_is_rejected(self):
        row = self.run_stream([
            {"choices": [{"delta": {"content": "391"}, "finish_reason": "stop"}]},
            {"usage": {"completion_tokens": 1}},
        ], done=False)
        self.assertIsNotNone(row["error"])
        self.assertFalse(row["done"])

    def test_stream_error_is_rejected(self):
        row = self.run_stream([{"error": {"message": "engine failed"}}])
        self.assertIn("engine failed", row["error"])


if __name__ == "__main__":
    unittest.main()
