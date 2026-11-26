# Reference: `apertux.sh` Configuration

## Subcommands

| Subcommand | Description |
| --- | --- |
| `up` | Default action. Runs preflight, 200GbE detection, local-only cache check, optional quantization + sync, Ray + vLLM startup, health polling, and smoke test. |
| `quantize` | Runs offline 128 GB-safe `llm-compressor` quantization (`nvfp4` or `fp8`) into `QUANT_BASE_DIR` and syncs the checkpoint to `--worker`. |
| `bench` | Runs a deterministic generation prompt via `curl` and reports `completion_tokens`, elapsed time, and `tok/s`. |
| `status` | Checks `apertux-node` container state, Ray GPU registration, HTTP `/health`, and recent vLLM log lines. |
| `smoke` | Sends a `POST /v1/chat/completions` test via `curl` and verifies a valid JSON completion. |
| `logs` | Prints recent `/tmp/vllm-serve.log` output from `apertux-node` (`--follow` streams live). |
| `down` | Stops and removes `apertux-node` locally and on `--worker` (if specified). |

## CLI Flags and Environment Variables

| Flag | Environment Variable | Default | Description |
| --- | --- | --- | --- |
| `--turbo` | - | `(off)` | Preset enabling `--quant nvfp4 --spec-mode draft` (unless overridden). |
| `--quant <scheme>` | `QUANT_SCHEME` | `none` | `none` (`bf16`), `fp8` (`FP8_DYNAMIC`), or `nvfp4` (`NVFP4`). |
| `--quant-dir <dir>` | `QUANT_BASE_DIR` | `$HOME/.cache/apertux/quantized` | Host base directory for quantized checkpoints on both Sparks. |
| `--calib-file <jsonl>` | `CALIB_FILE` | `(built-in)` | Local JSONL calibration file for `nvfp4` quantization. |
| `--calib-samples <int>` | `CALIB_SAMPLES` | `128` | Number of calibration sequences for `nvfp4`. |
| `--force-quant` | `FORCE_QUANT` | `0` | Force re-quantization and re-sync even when `apertux_quant_meta.json` exists. |
| `--spec-mode <mode>` | `SPEC_MODE` | `off` | `off`, `draft` (secondary small Apertus model), or `ngram` (prompt lookup). |
| `--draft-model <id>` | `DRAFT_MODEL` | `swiss-ai/Apertus-v1.1-1.5B-Instruct` | Draft checkpoint used when `--spec-mode draft`. |
| `--spec-tokens <int>` | `SPEC_TOKENS` | `5` | Candidate tokens proposed per speculative step. |
| `--prompt-lookup-max <n>` | `PROMPT_LOOKUP_MAX` | `4` | Maximum n-gram lookup window for `--spec-mode ngram`. |
| `--max-tokens <int>` | `BENCH_MAX_TOKENS` | `256` | Completion budget for `apertux.sh bench`. |
| `--worker <user@host>` | `WORKER_SSH` | `(empty)` | SSH destination for Spark 2 in cluster mode. |
| `--model <id-or-path>` | `MODEL` | `swiss-ai/Apertus-v1.5-70B` | Hugging Face model ID or local `.safetensors` directory. |
| `--image <image>` | `VLLM_IMAGE` | `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64` | ARM64 container image with CUDA 13 (`12.0+PTX`) and Apertus 1.5 support. |
| `--fabric-iface <if>` | `FABRIC_IFACE` | `(auto)` | ConnectX-7 200GbE interface on local Spark. |
| `--assign-fabric-ips` | `ASSIGN_FABRIC_IPS` | `0` | Assign `192.168.100.1/24` (head) and `192.168.100.2/24` (worker) at runtime if missing. |
| `--max-model-len <int>` | `MAX_MODEL_LEN` | `262144` | Maximum sequence length in tokens. |
| `--kv-cache-dtype <dtype>` | `KV_CACHE_DTYPE` | `fp8` | KV cache precision (`fp8`, `fp8_e4m3`, `fp8_e5m2`, `auto`). |
| `--no-prefix-caching` | `ENABLE_PREFIX_CACHING` | `1` | Set `ENABLE_PREFIX_CACHING=0` to disable automatic prefix caching. |
| `--gpu-mem-util <float>` | `GPU_MEM_UTIL` | `0.88` | Fraction of per-node 128 GB unified memory reserved by vLLM. |
| `--mode <mode>` | `SERVE_MODE` | `tools` | `tools`, `thinking`, or `plain`. |
| `--port <int>` | `API_PORT` | `8000` | OpenAI-compatible HTTP listen port on Spark 1. |
| `--hf-cache <dir>` | `HF_CACHE_DIR` | `$HOME/.cache/huggingface` | Host Hugging Face cache directory. |
| `--nccl-ib-disable` | `NCCL_IB_DISABLE` | `0` | Force TCP socket transport over 200GbE instead of RoCE/IB. |
| `--approve-download` | `APPROVE_DOWNLOAD` | `0` | Authorize first-time image, weight, or wheel cache population. |
| `--dry-run` | - | `0` | Print resolved topology and `vllm serve` command without starting containers. |

## Throughput and Memory Tiers on 2x DGX Spark (`TP=2`, 546 GB/s Pooled LPDDR5X)

| Configuration | Total Weight Footprint | Free Memory for `fp8` KV Cache | Expected Single-Stream Speed |
| --- | --- | --- | --- |
| `bfloat16` (default) | ~145 GB (~72.5 GB/Spark) | ~76 GB | ~2.5 - 3.5 tok/s |
| `--quant fp8` (`FP8_DYNAMIC`) | ~70 GB (~35 GB/Spark) | ~150 GB | ~5.5 - 7.0 tok/s |
| `--quant nvfp4` (`NVFP4`) | ~38 GB (~19 GB/Spark) | ~180 GB | ~10 - 13 tok/s |
| `--turbo` (`NVFP4` + `draft` / `ngram` specdec) | ~41 GB (incl. 1.5B draft) | ~175 GB | ~20 - 30 tok/s |
