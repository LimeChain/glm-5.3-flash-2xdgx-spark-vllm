#!/usr/bin/env python3
"""Two-rank NCCL all-reduce microbenchmark in the serving image."""
import datetime
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

torch.cuda.set_device(0)
rank = int(os.environ["RANK"])
backend = os.getenv("BENCH_BACKEND", "torch")
dist.init_process_group("gloo" if backend == "pynccl" else "nccl", timeout=datetime.timedelta(seconds=90))
comm = None
if backend == "pynccl":
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    comm = PyNcclCommunicator(dist.group.WORLD, device=0)
    assert comm.available and not comm.disabled


def all_reduce(tensor):
    if comm is None:
        dist.all_reduce(tensor)
    else:
        comm.all_reduce(tensor, tensor, stream=torch.cuda.current_stream())


rows = []
for nbytes in (16384, 65536, 131072, 262144, 524288, 1048576, 16777216, 67108864):
    tensor = torch.full((nbytes // 2,), rank + 1, dtype=torch.bfloat16, device="cuda")
    all_reduce(tensor)
    torch.cuda.synchronize()
    assert torch.all(tensor == 3).item(), "all-reduce correctness failure"
    tensor.zero_()
    for _ in range(20):
        all_reduce(tensor)
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            all_reduce(tensor)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) / 100)
    seconds = statistics.median(samples)
    row = {"bytes": nbytes, "median_us": seconds * 1e6, "algorithm_GB_s": nbytes / seconds / 1e9, "round_seconds": samples}
    if os.getenv("BENCH_GRAPHS") == "1":
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(20):
                all_reduce(tensor)
        graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(3):
            dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) / 100)
        graph_s = statistics.median(samples)
        row.update(graph_median_us=graph_s * 1e6, graph_algorithm_GB_s=nbytes / graph_s / 1e9)
        del graph
        torch.cuda.synchronize()
    rows.append(row)
if rank == 0:
    result = json.dumps({"torch": torch.__version__, "nccl": torch.cuda.nccl.version() if comm is None else comm.nccl.ncclGetVersion(), "backend": backend, "dtype": "bfloat16", "hca": os.getenv("NCCL_IB_HCA"), "protocol": os.getenv("NCCL_PROTO"), "channels": os.getenv("NCCL_MAX_NCHANNELS"), "rows": rows})
    if os.getenv("BENCH_OUTPUT"):
        Path(os.environ["BENCH_OUTPUT"]).write_text(result + "\n")
    else:
        print(result, flush=True)
dist.barrier()
if comm is not None and hasattr(comm.nccl, "ncclCommDestroy"):
    comm.nccl.ncclCommDestroy(comm.comm)
dist.destroy_process_group()
