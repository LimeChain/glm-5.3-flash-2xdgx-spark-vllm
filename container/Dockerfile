ARG BASE_IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce
FROM ${BASE_IMAGE}

ARG REPOSITORY_URL=https://github.com/LimeChain/glm-5.3-flash-2xdgx-spark-vllm
ARG REVISION=uncommitted
ARG BASE_IMAGE_DIGEST=sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce

LABEL org.opencontainers.image.title="GLM-5.3 Flash NVFP4 runtime for GB10" \
      org.opencontainers.image.description="vLLM GLM-5.3 NoPE sparse-MLA compatibility overlay for NVIDIA GB10 / SM121" \
      org.opencontainers.image.source="${REPOSITORY_URL}" \
      org.opencontainers.image.revision="${REVISION}" \
      io.glm53-gb10.base-image-digest="${BASE_IMAGE_DIGEST}" \
      io.glm53-gb10.logical-mla="nope256-rope0-kv512-v256" \
      io.glm53-gb10.physical-mla="glm-nsa-656b-pad64" \
      io.glm53-gb10.tp-local-heads="32,64" \
      io.glm53-gb10.cache-block-size="256"

COPY overlay/sitecustomize.py /opt/glm53-gb10-overlay/sitecustomize.py
COPY scripts/verify-runtime-contract.py /opt/glm53-gb10-overlay/verify-runtime-contract.py
COPY container/flashinfer-glm53.patch /opt/glm53-gb10-overlay/flashinfer-glm53.patch

# Add the exact TP1/TP2 decode+prefill specializations required by GLM-5.3's
# kpool buffer: 2048 selected history tokens + 3 tail slots, rounded to 2176
# for BLOCK_N=128.
RUN patch --batch --forward -p0 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /opt/glm53-gb10-overlay/flashinfer-glm53.patch

# The base image ships this FlashInfer module as an AOT .so, which takes
# precedence over edited JIT sources. Rebuild it for GB10/SM121a at image-build
# time and install the result back into the same AOT slot.
RUN mv \
      /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so \
      /opt/glm53-gb10-overlay/sparse_mla_sm120.base.so \
    && FLASHINFER_CUDA_ARCH_LIST=12.1a python3 -c \
      'from flashinfer.jit.mla import gen_sparse_mla_sm120_module; gen_sparse_mla_sm120_module().build()' \
    && install -m 0755 \
      /root/.cache/flashinfer/0.6.17/121a/cached_ops/sparse_mla_sm120/sparse_mla_sm120.so \
      /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so

ENV PYTHONPATH=/opt/glm53-gb10-overlay

RUN python3 -m py_compile \
      /opt/glm53-gb10-overlay/sitecustomize.py \
      /opt/glm53-gb10-overlay/verify-runtime-contract.py \
    && GLM53_GB10_PATCH_DISABLE=1 \
      python3 /opt/glm53-gb10-overlay/verify-runtime-contract.py

# Preserve the base image ENTRYPOINT ["vllm", "serve"].
