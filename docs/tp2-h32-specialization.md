# TP2 local-head specialization for GLM-5.3

Production source revision: `4c290638cb174721217c903e2dbf92b9858a080b`.

Base adapter source: [`cyijun/glm-5.3-flash-nvfp4-gb10`](https://github.com/cyijun/glm-5.3-flash-nvfp4-gb10) at `5bb0a598829839a9e0c420c6b737a742e084948c`.

## Failure that identified the gap

A dual-GB10 TP2 launch loaded the complete `LibertAIDAI/GLM-5.3-Flash-NVFP4` checkpoint on both ranks, then failed during mixed prefill/decode warm-up:

```text
Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2
or sparse_mla_sm120_decode_dsv4; got num_tokens=2
```

The model has 64 query heads globally. TP2 presents 32 local query heads to each rank. The initial adapter instantiated GLM's physical top-k capacity (`2176`) only for 64 heads, covering the single-rank fixture but not TP2.

The same gap existed in both paths:

- decode: `(num_heads=32, topk=2176)` was absent from the Python dispatch set and AOT module;
- prefill: `GLM_NSA, num_heads=32, topk=2176, page_block_size=64` was absent from the AOT prefill module.

## Minimal production patch

The patch adds exactly the missing TP2-local specializations while retaining the TP1 controls:

- decode: `DSV3_2_DISPATCH(32, 2176)`;
- prefill: `launch_prefill_mg<GLM_NSA, FP8, 32, 2176, 64>`;
- controls: the equivalent H=64/top-k=2176 decode and prefill paths.

The Docker build patches both source dispatch tables, rebuilds the FlashInfer SM121a AOT module, replaces the precompiled module, and verifies that the replacement differs from the base binary.

## Qualified result

The resulting image passed:

- full approximately 181 GiB checkpoint load on two DGX Sparks;
- H32 decode and prefill warm-up;
- deterministic API and tool-call checks;
- a 140,012-token prompt;
- MTP3 at 68.06% observed acceptance;
- matched C1/C4/C6/C8 throughput up to 101.74 aggregate tok/s at C8;
- zero restarts and zero OOM kills during qualification.

See [`test-matrix.md`](test-matrix.md) and the machine-readable benchmark receipt under `results/`.
