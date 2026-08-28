#!/usr/bin/env python3
"""Fail the image build if the pinned base runtime drifts underneath the overlay."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import inspect
from pathlib import Path

import flashinfer.mla._sparse_mla_sm120 as sparse_sm120
from vllm.v1.attention.backend import MLAAttentionImpl
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)


expected_versions = {
    "vllm": "0.1.dev20051+g487ecf187",
    "flashinfer-python": "0.6.17",
    "transformers": "5.15.1",
}
for package, expected in expected_versions.items():
    actual = importlib.metadata.version(package)
    assert actual == expected, f"{package}: expected {expected}, got {actual}"

assert FlashInferMLASparseSM120Backend.get_supported_head_sizes() == [512, 576]
assert 256 in FlashInferMLASparseSM120Backend.get_supported_kernel_block_sizes()
assert "concat_and_cache_mla" in inspect.getsource(MLAAttentionImpl.do_kv_cache_update)
assert hasattr(sparse_sm120, "_decode_dsv3_2_dispatchable")
for local_heads in (32, 64):
    assert (local_heads, 2176) in sparse_sm120._DECODE_DSV3_2_DISPATCH
assert hasattr(FlashInferMLASparseSM120Impl, "forward_mqa")

flashinfer_spec = importlib.util.find_spec("flashinfer")
assert flashinfer_spec is not None and flashinfer_spec.origin is not None
flashinfer_root = Path(flashinfer_spec.origin).parent
decode_source = flashinfer_root / "data/csrc/sparse_mla_sm120_decode_dsv3_2.cu"
prefill_source = flashinfer_root / "data/csrc/sparse_mla_sm120_prefill.cu"
decode_text = decode_source.read_text()
prefill_text = prefill_source.read_text()
for local_heads in (32, 64):
    assert f"DSV3_2_DISPATCH({local_heads}, 2176)" in decode_text
    assert f"ComputeMode::FP8, {local_heads}, 2176, 64" in prefill_text

jit_cache_spec = importlib.util.find_spec("flashinfer_jit_cache")
assert jit_cache_spec is not None and jit_cache_spec.origin is not None
aot_module = (
    Path(jit_cache_spec.origin).parent
    / "jit_cache/sparse_mla_sm120/sparse_mla_sm120.so"
)
base_module = Path("/opt/glm53-gb10-overlay/sparse_mla_sm120.base.so")
assert aot_module.is_file() and base_module.is_file()
assert hashlib.sha256(aot_module.read_bytes()).digest() != hashlib.sha256(
    base_module.read_bytes()
).digest(), "rebuilt AOT module is identical to the base artifact"

print("base runtime and rebuilt FlashInfer AOT contract verified")
