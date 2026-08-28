# Adaptation design

## Kernel- and layout-selecting dimensions

The faithful mock reduces ordinary width and depth, but keeps the following
target values because they select execution branches, CUDA specializations,
packed weights, or cache layouts:

| Area | Fixture and target value |
| --- | ---: |
| layer pattern | 3 KDA + 1 sparse MLA |
| attention / KV heads | 64 / 64 |
| Q LoRA / KV LoRA rank | 1536 / 512 |
| QK NoPE / RoPE | 256 / 0 |
| V head dimension | 256 |
| KDA heads / head dimension / convolution | 64 / 128 / 4 |
| index heads / head dimension | 32 / 128 |
| index top-k / k-pool | 2048 / 4 |
| routed / selected / shared experts | 288 / 8 / 1 |
| MoE / dense intermediate | 2048 / 12288 |
| dense prefix | first 3 layers |
| MTP layers | 1 |
| vocabulary | 154880 |

The fixture uses four base layers plus one MTP copy; the real model uses 45
base layers. The fixture's hidden size is 256 instead of 4096, maximum context
is 8192 instead of 1,048,576, and its vision tower is deliberately tiny. Those
reductions do not bypass any compatibility path targeted by this adapter.

Its routed-expert tensors keep the target ModelOpt NVFP4 ABI: packed `U8`
weights, `F8_E4M3` block scales, and scalar `F32` global scales. All 1,728
expert projections required by the reduced layer stack are present.

## Two cache block sizes, for two layers

The vLLM CLI block size must be divisible by `index_kpool * 32`. With k-pool 4,
the dedicated image's supported choice is `--block-size 256`.

That is not the page size passed to the compressed FlashInfer sparse-attention
kernel. vLLM's internal metadata presents that physical cache with 64-token
pages. The overlay checks for 64 at the FlashInfer call boundary; it does not
widen a native page-size dispatch guard to 256.

## NoPE to GLM_NSA physical ABI

The vLLM path absorbs the logical 256-wide query through the MLA key
up-projection, producing a 512-wide latent query. FlashInfer's GLM_NSA kernel
consumes a 576-wide query and a 656-byte packed cache row:

```text
physical query = [absorbed_query_512 | zeros_64]
physical KV    = [latent_fp8_512 | scales_fp32_4 | zeros_bf16_64]
```

The adapter passes the original attention scale. Its physical
`qk_rope_head_dim=64` argument is an ABI selector only; the model remains
logically `qk_rope_head_dim=0`.

The converted sparse indices retain a valid-count tensor per row. Empty rows
are temporarily pointed to slot zero with length one and their resulting output
is masked to zero, avoiding an invalid native launch without attending to
padding.

## Why top-k 2048 requires a 2176 specialization

`index_topk=2048` describes the architectural history selection. GLM's k-pool
path adds an always-selected tail of up to three tokens, and the downstream
buffer is rounded to FlashInfer's `BLOCK_N=128`. The native module therefore
receives a capacity of 2176:

```text
round_up(2048 + 3, 128) = 2176
```

The bundled AOT module contains `topk=2048`, not 2176. TP1 needs an exact
`(64, 2176)` instantiation; TP2 presents 32 local heads to each rank and needs
`(32, 2176)`. More than 64 query tokens enter the prefill orchestrator, so
matching `GLM_NSA` prefill instantiations are required as well.

The installed AOT `.so` takes precedence over the editable JIT sources. The
Docker build patches both source dispatch tables, compiles for `12.1a`, and
replaces the installed AOT artifact. The original `.so` is retained in the
image for audit comparison.

## Overlay points

1. `MLAAttentionImpl.do_kv_cache_update`: supply 64 physical BF16 zeros when
   the logical `k_pe` width is zero and the cache dtype is `fp8_ds_mla`.
2. `FlashInferMLASparseSM120Impl.forward_mqa`: validate the GLM contract, pad
   the query, convert sparse indices with valid counts, and call the physical
   GLM_NSA ABI without changing model configuration.
3. FlashInfer AOT decode and prefill dispatch: add the exact H=32 and H=64,
   top-k=2176 specializations and rebuild for SM121a.

## Why the DeepSeek-V4 584-byte path is not interchangeable

The earlier DeepSeek-V4 GB10 work is useful for build and TP2 orchestration,
but its cache stores 448 FP8 NoPE bytes, 64 BF16 RoPE values, and UE8M0 scale
footers (584 bytes/token). GLM-5.3 uses a 512-value latent, four arbitrary FP32
scales, no logical RoPE, and the 2176 sparse-buffer capacity. Reusing the DSV4
kernel would change the quantization/cache contract and still lack the required
top-k specialization.
