import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "bench/benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class Regressions(unittest.TestCase):
    def test_missing_metrics_are_not_reported_as_idle_zero_counters(self):
        with patch.object(benchmark.urllib.request, "urlopen", return_value=io.BytesIO(b"vllm:num_requests_running 0\n")):
            with self.assertRaisesRegex(RuntimeError, "missing required vLLM metrics"):
                benchmark.metric_snapshot("http://localhost")

    def test_failed_start_stops_both_attempted_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("scripts", "config", "bin"):
                (root / name).mkdir()
            shutil.copy2(ROOT / "scripts/start-tp2.sh", root / "scripts/start-tp2.sh")
            (root / "config/cluster.env").write_text("WORKER_SSH=mock\nREMOTE_ROOT=/mock\nCONTAINER_NAME=test-glm\nAPI_PORT=8888\n")
            scripts = {
                "scripts/preflight-tp2.sh": "exit 0",
                "scripts/rank-tp2.sh": "exit 0",
                "bin/ssh": 'echo "ssh $*" >> "$CALL_LOG"\ncase "$*" in *"docker ps"*) exit 1;; *"docker inspect"*) echo "false|false|1";; esac',
                "bin/docker": 'echo "docker $*" >> "$CALL_LOG"\ncase "$1" in inspect) echo "false|false|1";; esac',
                "bin/curl": "exit 1",
                "bin/sleep": "exit 0",
            }
            for name, body in scripts.items():
                script = root / name
                script.write_text("#!/bin/bash\n" + body + "\n")
                script.chmod(0o755)
            log = root / "calls.log"
            env = dict(os.environ, PATH=str(root / "bin") + ":" + os.environ["PATH"], CALL_LOG=str(log))
            env.pop("CONFIG_FILE", None)
            result = subprocess.run(["bash", str(root / "scripts/start-tp2.sh")], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 12, result.stderr)
            calls = log.read_text().splitlines()
            self.assertIn("docker stop --time 60 test-glm", calls)
            self.assertTrue(any(line.startswith("ssh ") and "docker stop --time 60 'test-glm'" in line for line in calls), calls)

    def test_preflight_checks_both_ranks_without_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "config").mkdir()
            (root / "bin").mkdir()
            shutil.copy2(ROOT / "scripts/preflight-tp2.sh", root / "scripts/preflight-tp2.sh")
            (root / "config/cluster.env").write_text("WORKER_SSH=mock\nREMOTE_ROOT=/mock\n")
            rank = root / "scripts/rank-tp2.sh"
            rank.write_text('#!/bin/bash\n[[ "$PREFLIGHT_ONLY" == 1 ]] || exit 99\necho "preflight_ok image=abc revision=def profile=ghi model_meta=jkl"\n')
            rank.chmod(0o755)
            ssh = root / "bin/ssh"
            ssh.write_text('#!/bin/bash\ncase "$*" in *PREFLIGHT_ONLY=1*) echo "preflight_ok image=abc revision=def profile=ghi model_meta=jkl";; esac\n')
            ssh.chmod(0o755)
            env = dict(os.environ, PATH=str(root / "bin") + ":" + os.environ["PATH"])
            env.pop("PREFLIGHT_ONLY", None)
            env.pop("CONFIG_FILE", None)
            result = subprocess.run(["bash", str(root / "scripts/preflight-tp2.sh")], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("TP2_PREFLIGHT_OK", result.stdout)
            ssh.write_text(ssh.read_text().replace("image=abc", "image=mismatch"))
            result = subprocess.run(["bash", str(root / "scripts/preflight-tp2.sh")], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("parity mismatch", result.stderr)

    def test_stream_error_is_not_a_success(self):
        class Stream:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                event = {"system_fingerprint": "test", "choices": [{"delta": {"content": "1"}, "finish_reason": "length"}], "usage": {"completion_tokens": 512}}
                yield ("data: " + json.dumps(event)).encode()
                raise OSError("simulated interrupted stream")
        with patch.object(benchmark.urllib.request, "urlopen", return_value=Stream()):
            row = benchmark.stream_one("http://localhost", {}, "test", threading.Barrier(1))
        self.assertIsNotNone(row["error"])
        self.assertFalse(row["stream_done"])
        wave = {"label": "measured-1", "rows": [row], "summary": {"aggregate_active_decode_tokens_per_second": 1, "median_per_stream_decode_tokens_per_second": 1, "median_ttft_seconds": 1, "peak_running": 1, "peak_waiting": 0, "system_fingerprints": ["test"], "metric_delta": {k: 0 for k in benchmark.METRICS}}}
        self.assertFalse(benchmark.summarize_scenario([wave], 512)["all_http_success"])


if __name__ == "__main__":
    unittest.main()
