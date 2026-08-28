# GLM-5.3 Flash NVFP4：双 NVIDIA DGX Spark / TP2

[English](README.md)

本仓库提供在两台 NVIDIA DGX Spark（GB10 / SM121）上运行完整 **GLM-5.3 Flash NVFP4** 模型所需的 vLLM / FlashInfer 兼容层和 TP2 启动脚本。

## 实测性能

生产配置：TP2、MTP3、FlashInfer sparse MLA、FP8 KV cache、Marlin NVFP4、262,144 上下文配置、eager execution。

| 并发 | 聚合 decode 吞吐 | 最佳 cold-run median |
|---:|---:|---:|
| C1 | **29.74 tok/s** | 30.11 tok/s |
| C4 | **68.07 tok/s** | 68.82 tok/s |
| C6 | **87.39 tok/s** | 87.43 tok/s |
| C8 | **101.74 tok/s** | 102.34 tok/s |

C4/C6/C8 是**聚合吞吐**，不是单请求速度。每个场景使用一次 warm-up 和三次 512-token 测量；表中主结果是两次独立 cold start 的 median 平均值。完整数据见 [`results/production-tp2-mtp3-262k.json`](results/production-tp2-mtp3-262k.json)。

## 核心修改

GLM-5.3 逻辑上是 NoPE，但当前 FlashInfer GLM_NSA kernel 的物理 ABI 需要：

```text
query = [吸收后的 NoPE 512 | 64 个零]
KV    = [FP8 latent 512 | 4 个 FP32 scale | 64 个 BF16 零]
```

模型的 `index_topk=2048` 经过 tail 和 `BLOCK_N=128` 对齐后，实际 sparse buffer 宽度是 2176。TP2 每个 rank 看到 32 个本地 query heads，因此本仓库增加 `(32, 2176)` 和 `(64, 2176)` 的 decode/prefill 特化，并为 `sm_121a` 重新编译 FlashInfer AOT 模块。

## 快速开始

在两台节点上构建同一镜像：

```bash
git clone https://github.com/LimeChain/glm-5.3-flash-2xdgx-spark-vllm.git
cd glm-5.3-flash-2xdgx-spark-vllm
IMAGE=glm53-sm121:local ./scripts/build-image.sh
```

在 head 节点配置并启动：

```bash
cp config/cluster.env.example config/cluster.env
$EDITOR config/cluster.env
./scripts/preflight-tp2.sh
./scripts/start-tp2.sh
```

默认 API 只监听 head 的 loopback：`http://127.0.0.1:8000`。

完整说明、验证边界和许可证状态以英文 [`README.md`](README.md) 为准。
