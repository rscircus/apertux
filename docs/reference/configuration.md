# Reference: `apertux.sh` Configuration

## Subcommands

| Subcommand | Description |
| --- | --- |
| `up` | Default action. Runs preflight, 200GbE fabric detection, local-only cache check, Ray + vLLM startup, health polling, and smoke test. |
| `status` | Checks `apertux-node` container state, Ray GPU registration, HTTP `/health`, and recent vLLM log lines. |
| `smoke` | Sends a deterministic `POST /v1/chat/completions` test via `curl` and verifies a valid JSON completion. |
| `logs` | Prints recent `/tmp/vllm-serve.log` output from `apertux-node` (`--follow` streams live). |
| `down` | Stops and removes `apertux-node` locally and on `--worker` (if specified). |

## CLI Flags and Environment Variables

| Flag | Environment Variable | Default | Description |
| --- | --- | --- | --- |
| `--worker <user@host>` | `WORKER_SSH` | `(empty)` | SSH destination for Spark 2 in cluster mode. |
| `--head` | - | `cluster` | Run Spark 1 head node without SSH dispatch to Spark 2. |
| `--worker-only` | - | `cluster` | Run Spark 2 Ray worker container only (requires `--head-ip`). |
| `--head-ip <ipv4>` | `HEAD_IP` | `(empty)` | Spark 1 200GbE IPv4 address for `--worker-only`. |
| `--model <id-or-path>` | `MODEL` | `swiss-ai/Apertus-v1.5-70B` | Hugging Face model ID or local `.safetensors` directory. |
| `--image <image>` | `VLLM_IMAGE` | `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64` | ARM64 container image with CUDA 13 (`12.0+PTX`) and Apertus 1.5 support. |
| `--fabric-iface <if>` | `FABRIC_IFACE` | `(auto)` | ConnectX-7 200GbE interface on local Spark. |
| `--worker-fabric-iface <if>` | `WORKER_FABRIC_IFACE` | `(auto)` | ConnectX-7 200GbE interface on Spark 2. |
| `--assign-fabric-ips` | `ASSIGN_FABRIC_IPS` | `0` | Assign `192.168.100.1/24` (head) and `192.168.100.2/24` (worker) at runtime if missing. |
| `--max-model-len <int>` | `MAX_MODEL_LEN` | `262144` | Maximum sequence length in tokens. |
| `--kv-cache-dtype <dtype>` | `KV_CACHE_DTYPE` | `fp8` | KV cache precision (`fp8`, `fp8_e4m3`, `fp8_e5m2`, `auto`). |
| `--no-prefix-caching` | `ENABLE_PREFIX_CACHING` | `1` | Set `ENABLE_PREFIX_CACHING=0` to disable automatic prefix caching. |
| `--gpu-mem-util <float>` | `GPU_MEM_UTIL` | `0.88` | Fraction of per-node 128 GB unified memory reserved by vLLM. |
| `--mode <mode>` | `SERVE_MODE` | `tools` | `tools` (`--tool-call-parser apertus`), `thinking` (`--reasoning-parser apertus`), or `plain`. |
| `--port <int>` | `API_PORT` | `8000` | OpenAI-compatible HTTP listen port on Spark 1. |
| `--hf-cache <dir>` | `HF_CACHE_DIR` | `$HOME/.cache/huggingface` | Host Hugging Face cache directory mounted into `/root/.cache/huggingface`. |
| `--nccl-ib-disable` | `NCCL_IB_DISABLE` | `0` | Force TCP socket transport over the 200GbE interface instead of RoCE/IB. |
| `--approve-download` | `APPROVE_DOWNLOAD` | `0` | Authorize first-time container pull and weight download without interactive `/dev/tty` prompt. |
| `--dry-run` | - | `0` | Print resolved topology and `vllm serve` command without starting containers. |

## Network Ports and Pinned Interfaces

| Traffic | Port | Interface Binding |
| --- | --- | --- |
| OpenAI HTTP API | `8000/tcp` (`API_PORT`) | `0.0.0.0` (`API_HOST`) on Spark 1 |
| Ray GCS head | `6379/tcp` (`RAY_PORT`) | `HEAD_FABRIC_IP` (200GbE ConnectX-7 only) |
| NCCL / Gloo / UCX / Tensor Parallel | Ephemeral (`--network host`) | Pinned via `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `UCX_NET_DEVICES` |

## Memory and Bandwidth Profile (2x DGX Spark GB10)

| Metric | Single Spark | Dual Spark (`TP=2`) |
| --- | --- | --- |
| Unified LPDDR5X capacity | 128 GB | 256 GB (~240 GB usable after OS) |
| Memory bandwidth | 273 GB/s | 546 GB/s aggregate |
| `Apertus-v1.5-70B` `bfloat16` weights | Does not fit (~145 GB) | ~72.5 GB / node |
| KV cache (`fp8`, `262144` max context, `0.88` util) | - | ~38 GB / node (~76 GB pooled) |
| Expected single-stream `bfloat16` decode speed | - | ~2.5 - 3.5 tok/s (bandwidth-bound) |
