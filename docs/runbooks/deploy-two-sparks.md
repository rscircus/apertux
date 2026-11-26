# Runbook: Deploy Apertus 70B on Two DGX Sparks

## Prerequisites

1. Two NVIDIA DGX Spark units (`aarch64` GB10, 128 GB unified memory each) running DGX OS with Docker and NVIDIA Container Toolkit installed.
2. One 200GbE QSFP112 DAC cable connected directly between a ConnectX-7 port on Spark 1 and a ConnectX-7 port on Spark 2.
3. Key-based SSH access from Spark 1 to Spark 2 (optional if using the two-terminal fallback).

## 1. Deploy from Spark 1 (`curl | bash`)

On Spark 1, run:

```bash
curl -fsSL https://tig.eigenlab.ch/roland/apertux/raw/branch/main/apertux.sh \
  | bash -s -- --worker nvidia@spark2 --assign-fabric-ips
```

What happens:
1. Verifies `aarch64`, Docker, and `nvidia-smi` on both Sparks.
2. Detects the `UP` ConnectX-7 200GbE interface on both Sparks, assigns `192.168.100.1/24` and `192.168.100.2/24` if missing, and verifies direct 200GbE ping.
3. Checks local Docker images and `~/.cache/huggingface`. On a cold cache, prompts on `/dev/tty` before downloading `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64` and `swiss-ai/Apertus-v1.5-70B` (pass `--approve-download` for non-interactive bootstrap). Warm caches start immediately with `HF_HUB_OFFLINE=1`.
4. Launches `apertux-node` on both Sparks, waits for `2.0` GPUs in Ray, starts `vllm serve` (`TP=2`, `262144` context, `fp8` KV cache, prefix caching), verifies `/health`, and runs a live `curl` completion check.

If you run from a local checkout or LAN mirror instead of `tig.eigenlab.ch`:

```bash
# Local checkout on Spark 1
./apertux.sh up --worker nvidia@spark2 --assign-fabric-ips

# Or serve apertux.sh over LAN from your laptop: python3 -m http.server 9000
curl -fsSL http://<laptop-ip>:9000/apertux.sh | bash -s -- --worker nvidia@spark2
```

## 2. Two-Terminal Fallback (No Cross-Node SSH)

If Spark 1 cannot SSH into Spark 2:

1. Assign static IPs on the active ConnectX-7 interface if not already set:
   - Spark 1: `sudo ip addr add 192.168.100.1/24 dev enp1s0f1np1 && sudo ip link set enp1s0f1np1 up`
   - Spark 2: `sudo ip addr add 192.168.100.2/24 dev enp1s0f1np1 && sudo ip link set enp1s0f1np1 up`
2. On Spark 2 (start worker first or within 60 seconds of head):
   ```bash
   curl -fsSL https://tig.eigenlab.ch/roland/apertux/raw/branch/main/apertux.sh \
     | bash -s -- --worker-only --head-ip 192.168.100.1
   ```
3. On Spark 1:
   ```bash
   curl -fsSL https://tig.eigenlab.ch/roland/apertux/raw/branch/main/apertux.sh \
     | bash -s -- --head
   ```

## 3. Query with `curl`

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "swiss-ai/Apertus-v1.5-70B",
    "messages": [{"role": "user", "content": "Hello from DGX Spark!"}],
    "max_tokens": 128
  }'
```

To run Apertus 1.5 in reasoning ("thinking") mode instead of tool-calling mode, redeploy with `--mode thinking`.

## 4. Inspect Status, Logs, and Teardown

```bash
./apertux.sh status --worker nvidia@spark2
./apertux.sh logs --follow
./apertux.sh smoke --endpoint http://127.0.0.1:8000
./apertux.sh down --worker nvidia@spark2
```

## 5. Troubleshooting

- **No active ConnectX-7 200GbE port found:** Run `ibdev2netdev` or `ip link` on both Sparks to confirm the QSFP cable shows `(Up)` / `LOWER_UP`, or specify `--fabric-iface <ifname>` explicitly.
- **Weight broadcast hangs in NCCL over RoCE:** Re-run `up` with `--nccl-ib-disable` to force TCP sockets over the 200GbE ConnectX-7 interface while keeping traffic off the 1GbE LAN.
- **OOM during startup:** Lower `--gpu-mem-util` (e.g. `0.82`) or reduce `--max-model-len` (e.g. `65536`) if other host processes hold unified LPDDR5X memory.
