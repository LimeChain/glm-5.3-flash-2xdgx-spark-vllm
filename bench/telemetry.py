#!/usr/bin/env python3
"""Record lightweight GPU and Linux unified-memory telemetry as JSON lines."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=int, default=7200)
parser.add_argument("--interval", type=float, default=5)
args = parser.parse_args()
if args.seconds <= 0 or args.interval <= 0:
    raise SystemExit("positive duration and interval required")
deadline = time.monotonic() + args.seconds
while time.monotonic() < deadline:
    sample = {"time": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname()}
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    sample["memory_kib"] = {key: int(memory[key].split()[0]) for key in ("MemAvailable", "MemFree", "SwapTotal", "SwapFree", "Cached")}
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    sample["vm_counters"] = {key: int(vm[key]) for key in ("pswpin", "pswpout", "pgmajfault")}
    sample["loadavg"] = Path("/proc/loadavg").read_text().split()[:3]
    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    sample["gpu"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    print(json.dumps(sample), flush=True)
    time.sleep(args.interval)
