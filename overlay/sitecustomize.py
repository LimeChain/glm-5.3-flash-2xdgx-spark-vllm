"""GLM-5.3 NoPE sparse-MLA compatibility overlay for SM120/SM121.

This file is deliberately a small, auditable runtime overlay. It must be
replaced by source-level upstream changes when those are available in the base
image. Set GLM53_GB10_PATCH_DISABLE=1 for static image inspection.
"""

from __future__ import annotations

import os


if os.getenv("GLM53_GB10_PATCH_DISABLE") != "1":
    import torch
    import flashinfer.mla._sparse_mla_sm120 as _sparse_mla_sm120

    from vllm.v1.attention.backend import MLAAttentionImpl
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        _get_workspace_buffer,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
        FlashInferMLASparseSM120Impl,
    )
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index,
    )

    required_glm_dispatch = {(32, 2176), (64, 2176)}
    missing_glm_dispatch = required_glm_dispatch.difference(
        _sparse_mla_sm120._DECODE_DSV3_2_DISPATCH
    )
    if missing_glm_dispatch:
        raise RuntimeError(
            "GLM-5.3 GB10 image is missing FlashInfer TP-local decode "
            f"specializations: {sorted(missing_glm_dispatch)}"
        )

    _original_cache_update = MLAAttentionImpl.do_kv_cache_update

    def _nope_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        # concat_and_cache_mla's fp8_ds_mla ABI requires pe_dim=64. GLM-5.3
        # has no positional component, so store physical zeros in that region.
        if kv_cache_dtype == "fp8_ds_mla" and k_pe.shape[-1] == 0:
            k_pe = k_pe.new_zeros((*k_pe.shape[:-1], 64))
        return _original_cache_update(
            self,
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    MLAAttentionImpl.do_kv_cache_update = _nope_cache_update

    def _nope_forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        layer,
    ) -> tuple[torch.Tensor, None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if self.qk_rope_head_dim != 0:
            raise RuntimeError("GLM-5.3 GB10 overlay received non-NoPE MLA")
        if q.shape[-1] != 512:
            raise RuntimeError(f"expected absorbed query width 512, got {q.shape[-1]}")
        if attn_metadata.block_size != 64:
            raise RuntimeError(
                "GLM-5.3 SM121 sparse MLA requires a physical attention page "
                f"of 64 tokens, got {attn_metadata.block_size}"
            )

        # Physical GLM_NSA ABI: [512 absorbed NoPE | 64 zero RoPE].
        q = torch.nn.functional.pad(q, (0, 64))
        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        if attn_metadata.topk_tokens != 2048:
            raise RuntimeError(
                "expected architecture index_topk=2048, got "
                f"{attn_metadata.topk_tokens}"
            )
        if topk_indices.shape[1] != 2176:
            raise RuntimeError(
                "expected the kpool/tail/alignment buffer width 2176, got "
                f"{topk_indices.shape[1]}"
            )

        topk_indices_physical, valid_counts = (
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        )
        empty_rows = valid_counts == 0
        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(
            empty_rows, 0
        )
        active_lengths = valid_counts.clamp(min=1)
        sparse_topk_capacity = topk_indices_physical.shape[1]

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            # This is a physical ABI selector, not a logical model mutation.
            qk_rope_head_dim=64,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=active_lengths,
            max_seq_len=sparse_topk_capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_topk_capacity,
            kv_scale_format=self.kv_scale_format,
        ).squeeze(1)
        out.masked_fill_(empty_rows[:, None, None], 0)
        return out, None

    FlashInferMLASparseSM120Impl.forward_mqa = _nope_forward_mqa
