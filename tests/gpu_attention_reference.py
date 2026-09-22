#!/usr/bin/env python3
"""Nonzero numerical regression for the packed GLM_NSA H32/H64 ABI."""
import json
import math

import torch
from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

torch.manual_seed(17)
device = "cuda"
capacity = 2176
kv = (torch.randn(capacity, 512, device=device) * .15).to(torch.float8_e4m3fn)
packed = torch.zeros(capacity, 656, dtype=torch.uint8, device=device)
packed[:, :512] = kv.view(torch.uint8)
scales = torch.tensor([.5, 1., 2., 4.], dtype=torch.float32, device=device).repeat(capacity, 1)
packed[:, 512:528] = scales.view(torch.uint8)
cache = packed.view(34, 64, 656)
results = []
for tokens in (1, 2, 64, 65):
    for heads in (32, 64):
        q = torch.zeros(tokens, heads, 576, dtype=torch.bfloat16, device=device)
        q[:, :, :512] = (torch.randn(tokens, heads, 512, device=device) * .1).to(torch.bfloat16)
        indices = torch.arange(capacity, dtype=torch.int32, device=device).repeat(tokens, 1)
        lengths = torch.tensor(([1, 17, 2048, 2051] * math.ceil(tokens / 4))[:tokens], dtype=torch.int32, device=device)
        indices.masked_fill_(torch.arange(capacity, device=device)[None, :] >= lengths[:, None], -1)
        output = torch.empty(tokens, heads, 512, dtype=torch.bfloat16, device=device)
        lse = torch.empty(tokens, heads, dtype=torch.float32, device=device)
        kwargs = {}
        if tokens <= 64:
            kwargs = {"mid_out": torch.empty(tokens, heads, 34, 512, dtype=torch.bfloat16, device=device), "mid_lse": torch.empty(tokens, heads, 34, dtype=torch.float32, device=device)}
        _sparse_mla_sm120_paged_attention(q, cache, indices, output, lse, .0625, d_v=512, kv_scale_format="arbitrary_fp32", topk_length=lengths, **kwargs)
        torch.cuda.synchronize()
        ref = torch.empty_like(output, dtype=torch.float32)
        latent = kv.float() * scales.repeat_interleave(128, dim=1)
        for row in range(tokens):
            valid = int(lengths[row])
            probabilities = torch.softmax(q[row, :, :512].float() @ latent[:valid].T * .0625, dim=-1)
            ref[row] = probabilities @ latent[:valid]
        error = (output.float() - ref).abs()
        max_abs = error.max().item()
        relative_l2 = (torch.linalg.vector_norm(output.float() - ref) / torch.linalg.vector_norm(ref)).item()
        assert torch.isfinite(output).all().item()
        assert max_abs < .01 and relative_l2 < .02, (tokens, heads, max_abs, relative_l2)
        results.append({"tokens": tokens, "heads": heads, "max_abs_error": max_abs, "relative_l2": relative_l2})
print(json.dumps({"pass": True, "results": results}))
