# apertux

Deploy Swiss-AI `Apertus-v1.5-70B` across two stacked NVIDIA DGX Sparks with a single `curl | bash` command and query it over an OpenAI-compatible HTTP endpoint.

## Primary Next Action

Run on **Spark 1** (with a 200GbE QSFP cable connected to Spark 2 and SSH access to `nvidia@spark2`):

```bash
curl -fsSL https://tig.eigenlab.ch/roland/apertux/raw/branch/main/apertux.sh \
  | bash -s -- --worker nvidia@spark2 --assign-fabric-ips
```

Once the built-in health and smoke check completes, query Apertus from any machine able to reach Spark 1:

```bash
curl -sS http://<spark1-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "swiss-ai/Apertus-v1.5-70B",
    "messages": [{"role": "user", "content": "Gruezi! Summarize the Swiss Federal Charter in two sentences."}],
    "max_tokens": 128
  }'
```

## Key Facts

1. **256 GB pooled unified memory (`TP=2`):** Shards `swiss-ai/Apertus-v1.5-70B` (`bfloat16`, ~145 GB weights) across both GB10 Grace Blackwell nodes with `--kv-cache-dtype fp8`, `--enable-prefix-caching`, and the full `262144` token context window.
2. **Strict 200GbE ConnectX-7 pinning:** Auto-detects active Mellanox/ConnectX-7 ports (`ibdev2netdev` / `enp1s0f*`) and pins Ray, Gloo, UCX, and NCCL traffic exclusively to the 200GbE link.
3. **Local-only governance:** Telemetry is disabled unconditionally; cached startups run strictly offline (`HF_HUB_OFFLINE=1`); first-time image or weight downloads pause for human confirmation on `/dev/tty` (or `--approve-download`).

## Overview

```mermaid
flowchart LR
    Operator["Operator Shell"] -->|"curl -fsSL tig.../apertux.sh | bash"| Spark1["Spark 1 (Head, GB10 128 GB)"]
    Spark1 -->|"SSH inline bash payload"| Spark2["Spark 2 (Worker, GB10 128 GB)"]
    Spark1 <-->|"200GbE ConnectX-7 QSFP (Ray + NCCL TP=2)"| Spark2
    Client["Client"] -->|"curl POST :8000/v1/chat/completions"| Spark1
```

## Documentation

- Runbook (how-to): [docs/runbooks/deploy-two-sparks.md](docs/runbooks/deploy-two-sparks.md)
- Reference (flags, env vars, ports, memory math): [docs/reference/configuration.md](docs/reference/configuration.md)
- Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
