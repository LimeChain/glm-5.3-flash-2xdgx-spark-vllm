# Validation matrix

Validated on 2026-08-27 on two NVIDIA DGX Spark systems (GB10 / SM121) with the full `LibertAIDAI/GLM-5.3-Flash-NVFP4` checkpoint.

| Gate | Status | Evidence/result |
|---|---|---|
| pinned base contract | pass | vLLM `0.1.dev20051+g487ecf187`, FlashInfer `0.6.17`, Transformers `5.15.1` |
| TP2 full-checkpoint load | pass | both ranks loaded the approximately 181 GiB checkpoint |
| local-head specializations | pass | H=32 and H=64, top-k capacity 2176, decode and prefill |
| rebuilt SM121a AOT module | pass | source dispatch and replacement binary contract verified |
| health and models APIs | pass | HTTP 200, expected model identity |
| deterministic arithmetic | pass | `17 × 23 = 391` |
| tool-call parser | pass | `multiply(a=17,b=23)` |
| MTP3 | pass | 68.06% observed speculative-token acceptance |
| matched throughput | pass | C1 29.74; C4 68.07 aggregate; C6 87.39 aggregate; C8 101.74 aggregate tok/s |
| configured context | pass | `max_model_len=262144` |
| completed long-context request | pass | 140,012 prompt tokens |
| realized KV capacity | pass | 372,827 tokens |
| stability during qualification | pass | zero restarts, zero OOM kills, zero post-ready kernel warnings |
| C12 admission | partial | 12 HTTP successes; peak 9 running and 3 waiting |
| C12 throughput | not measured | no C12 throughput claim |
| CUDA graph mode | not qualified | production profile uses eager execution |
| full 262K prompt | not measured | configured context is not presented as a completed 262K prompt |

The exact performance receipt is [`../results/production-tp2-mtp3-262k.json`](../results/production-tp2-mtp3-262k.json).

## Frozen production flags

```text
--tensor-parallel-size 2
--max-model-len 262144
--max-num-seqs 12
--max-num-batched-tokens 8192
--gpu-memory-utilization 0.82
--attention-backend FLASHINFER_MLA_SPARSE_SM120
--kv-cache-dtype fp8_ds_mla
--block-size 256
--moe-backend marlin
--enforce-eager
--speculative-config {"method":"mtp","num_speculative_tokens":3}
```

Every performance claim must bind to the model revision, source/image identity, workload, and exact profile. Re-run the benchmark after any material change.
