# Changelog

## [2026.11.26.1330] - 4f234db

- Added Dual-Chevron Spark X brand assets derived from Swiss-AI's `apertvs.svg` geometry: adaptive CSS-token SVGs (`assets/logo.svg`, `assets/icon.svg`), explicit light/dark wordmarks (`assets/logo-light.svg`, `assets/logo-dark.svg`), and `512x512` repo icon (`assets/icon.png`).
- Updated `README.md` with auto-switching `<picture>` header and added XML/accessibility/dimension assertions to `tests/test_apertux.sh`.

## [2026.11.26.1131] - 9262c9f

- Added `apertux.sh` single-command `curl | bash` orchestrator for serving `swiss-ai/Apertus-v1.5-70B` (`TP=2`, `bfloat16`, `262144` context, `fp8` KV cache, prefix caching) across two NVIDIA DGX Sparks over 200GbE ConnectX-7.
- Added ConnectX-7 200GbE auto-discovery (`ibdev2netdev` + sysfs fallback), inline SSH worker orchestration, standalone `--head` / `--worker-only` fallback, and `--assign-fabric-ips`.
- Enforced local-only cache governance (`HF_HUB_OFFLINE=1`, telemetry disabled, interactive `/dev/tty` or `--approve-download` gate for cold caches).
- Added startup traceback/OOM log inspection, `status`, `logs`, `down`, and `smoke` (`curl` `/v1/chat/completions`) subcommands, plus 94-assertion offline mock test suite in `tests/test_apertux.sh`.
- Added Diataxis runbook and configuration reference.
