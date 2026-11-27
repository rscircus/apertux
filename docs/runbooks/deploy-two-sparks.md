# Runbook: Deploy Apertus 70B on Two DGX Sparks

## Prerequisites

1. Two NVIDIA DGX Spark units (`aarch64` GB10, 128 GB unified memory each) running DGX OS with Docker and NVIDIA Container Toolkit installed.
2. One 200GbE QSFP112 DAC cable connected directly between a ConnectX-7 port on Spark 1 and a ConnectX-7 port on Spark 2.
3. Key-based SSH access from Spark 1 to Spark 2 (optional if using the two-terminal fallback).

## 1. Fast Interactive Deploy (`--turbo`: NVFP4 + Speculative Decoding, ~20+ tok/s)

On Spark 1, run:

```bash
curl -fsSL https://tig.eigenlab.ch/roland/apertux/raw/branch/apertux-turbo/apertux.sh \
  | bash -s -- --worker nvidia@spark2 --assign-fabric-ips --turbo
```

What `--turbo` does in addition to the baseline cluster checks:
1. Verifies local caches for both `swiss-ai/Apertus-v1.5-70B` and the draft model `swiss-ai/Apertus-v1.1-1.5B-Instruct` (prompting once on `/dev/tty` or using `--approve-download` if cold).
2. Runs one-time offline `NVFP4` quantization (`llm-compressor` with `max_memory={0: "28GiB", "cpu": "64GiB"}` + NVMe offload so 145 GB `bf16` weights never OOM a 128 GB Spark) into `~/.cache/apertux/quantized/` if not already built.
3. Streams the ~38 GB quantized checkpoint to Spark 2 over SSH (`tar`) so subsequent boots skip quantization and start directly.
4. Serves the `NVFP4` checkpoint at `TP=2` (`--dtype auto`, `262144` context, `fp8` KV cache, prefix caching) with `--speculative-config '{"method":"draft_model","model":"swiss-ai/Apertus-v1.1-1.5B-Instruct","num_speculative_tokens":5}'`.

To pre-build `FP8_DYNAMIC` (~70 GB, zero calibration pass) or `NVFP4` (~38 GB) separately before serving:

```bash
./apertux.sh quantize --quant fp8 --worker nvidia@spark2
./apertux.sh up --worker nvidia@spark2 --quant fp8 --spec-mode ngram
```

## 2. Baseline Unquantized `bfloat16` Deploy (~3 tok/s, Full Precision)

Omit `--turbo` to serve unquantized `bfloat16` `swiss-ai/Apertus-v1.5-70B` with `262144` context and `fp8` KV cache:

```bash
curl -fsSL https://tig.eigenlab.ch/roland/apertux/raw/branch/apertux-turbo/apertux.sh \
  | bash -s -- --worker nvidia@spark2 --assign-fabric-ips
```

## 3. Two-Terminal Fallback (No Cross-Node SSH)

1. Assign static IPs on the active ConnectX-7 interface if not already set:
   - Spark 1: `sudo ip addr add 192.168.100.1/24 dev enp1s0f1np1 && sudo ip link set enp1s0f1np1 up`
   - Spark 2: `sudo ip addr add 192.168.100.2/24 dev enp1s0f1np1 && sudo ip link set enp1s0f1np1 up`
2. On Spark 2:
   ```bash
   ./apertux.sh --worker-only --head-ip 192.168.100.1
   ```
3. On Spark 1:
   ```bash
   ./apertux.sh --head
   ```

## 4. Query with `curl` and Benchmark Throughput (`tok/s`)

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "swiss-ai/Apertus-v1.5-70B",
    "messages": [{"role": "user", "content": "Hello from DGX Spark!"}],
    "max_tokens": 128
  }'

# Measure single-stream decode speed in tok/s:
./apertux.sh bench --endpoint http://127.0.0.1:8000 --max-tokens 256
```

## 5. Inspect Status, Logs, and Teardown

```bash
./apertux.sh status --worker nvidia@spark2
./apertux.sh logs --follow
./apertux.sh down --worker nvidia@spark2
```

## 6. Troubleshooting

- **No active ConnectX-7 200GbE port found:** Run `ibdev2netdev` or `ip link` on both Sparks to confirm the QSFP cable shows `(Up)` / `LOWER_UP`, or pass `--fabric-iface <ifname>`.
- **Weight broadcast hangs in NCCL over RoCE:** Re-run `up` with `--nccl-ib-disable` to force TCP sockets over the 200GbE ConnectX-7 interface.
- **Custom NVFP4 calibration domain:** Pass `--calib-file /path/to/local.jsonl --force-quant` (lines of `{"text": "..."}` or `{"messages": [...]}`) to calibrate `NVFP4` scales on domain-specific local text offline.
