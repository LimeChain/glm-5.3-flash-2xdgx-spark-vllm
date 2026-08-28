# Third-party notices and license status

This repository is a source recipe and configuration kit. It does not include model weights, container layers, CUDA caches, or third-party binary distributions.

## Adapter provenance

The initial GLM-5.3 GB10 adapter source was based on:

- Repository: `https://github.com/cyijun/glm-5.3-flash-nvfp4-gb10`
- Base commit used for the TP2 adaptation: `5bb0a598829839a9e0c420c6b737a742e084948c`
- Production TP2 source revision: `4c290638cb174721217c903e2dbf92b9858a080b`

That source repository did not expose a repository-level license when this snapshot was prepared. No license is invented here. Files without an explicit license notice remain subject to their existing copyright status; this repository does not grant additional rights to them.

## Upstream projects

The runtime base and patch targets include or derive from:

- vLLM — Apache License 2.0: `https://github.com/vllm-project/vllm`
- FlashInfer — repository license Apache License 2.0, with path-specific notices including BSD-3-Clause: `https://github.com/flashinfer-ai/flashinfer`
- NVIDIA CUDA and container components — their respective NVIDIA terms.

All upstream copyright, SPDX, patent, trademark, and warranty notices remain authoritative.

## Model checkpoint

The model is external and must be acquired independently:

- `LibertAIDAI/GLM-5.3-Flash-NVFP4`
- Revision `9e0d74e3cef17f634e84fb8e2223707e02616290`
- The model repository identified its license as MIT when this snapshot was prepared.

This repository does not redistribute those model files or relicense them.

## No endorsement

Project and company names are used only for technical identification and attribution. No upstream endorsement is implied.
