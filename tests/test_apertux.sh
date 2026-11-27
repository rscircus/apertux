#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/apertux.sh"

PASS_COUNT=0
FAIL_COUNT=0

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  printf 'PASS: %s\n' "$1"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  printf 'FAIL: %s\n' "$1" >&2
  if [ "${2:-}" != "" ]; then
    printf '  detail: %s\n' "$2" >&2
  fi
}

assert_eq() {
  local expected="$1"
  local actual="$2"
  local msg="$3"
  if [ "$expected" = "$actual" ]; then
    pass "$msg"
  else
    fail "$msg" "expected=[$expected] actual=[$actual]"
  fi
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  local msg="$3"
  if printf '%s' "$haystack" | grep -Fq -- "$needle"; then
    pass "$msg"
  else
    fail "$msg" "missing [$needle] in output"
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  local msg="$3"
  if printf '%s' "$haystack" | grep -Fq -- "$needle"; then
    fail "$msg" "unexpected [$needle] found in output"
  else
    pass "$msg"
  fi
}

make_sandbox() {
  local tmp
  tmp="$(mktemp -d)"
  mkdir -p "$tmp/bin" "$tmp/state" "$tmp/hf/hub/models--swiss-ai--Apertus-v1.5-70B/snapshots/abc123"
  touch "$tmp/hf/hub/models--swiss-ai--Apertus-v1.5-70B/snapshots/abc123/model-00001-of-00030.safetensors"

  cat >"$tmp/bin/uname" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "-m" ]; then
  printf '%s\n' "${MOCK_ARCH:-aarch64}"
else
  printf 'Linux\n'
fi
EOF

  cat >"$tmp/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf 'GPU 0: NVIDIA GB10\n'
EOF

  cat >"$tmp/bin/ibdev2netdev" <<'EOF'
#!/usr/bin/env bash
if [ "${MOCK_NO_IBDEV:-0}" = "1" ]; then
  exit 127
fi
printf '%s\n' "${MOCK_IBDEV_OUT:-mlx5_0 port 1 ==> enp1s0f1np1 (Up)}"
EOF

  cat >"$tmp/bin/ip" <<'EOF'
#!/usr/bin/env bash
printf 'ip %s\n' "$*" >>"${MOCK_STATE_DIR}/commands.log"
if [ "${1:-}" = "-4" ] && [ "${2:-}" = "-o" ] && [ "${3:-}" = "addr" ] && [ "${4:-}" = "show" ]; then
  iface="${6:-${5:-}}"
  if [ "$iface" = "dev" ]; then
    iface="${6:-}"
  fi
  if [ -f "${MOCK_STATE_DIR}/ip_${iface}" ]; then
    cat "${MOCK_STATE_DIR}/ip_${iface}"
    exit 0
  fi
  if [ "${MOCK_HEAD_NO_IP:-0}" = "1" ]; then
    exit 0
  fi
  printf '3: %s    inet %s/24 brd 192.168.100.255 scope global %s\n' "$iface" "${MOCK_HEAD_IP:-192.168.100.1}" "$iface"
  exit 0
fi
if [ "${1:-}" = "addr" ] && [ "${2:-}" = "add" ]; then
  cidr="$3"
  iface="$5"
  ip_only="${cidr%/*}"
  printf '3: %s    inet %s brd 192.168.100.255 scope global %s\n' "$iface" "$cidr" "$iface" >"${MOCK_STATE_DIR}/ip_${iface}"
  exit 0
fi
if [ "${1:-}" = "link" ]; then
  iface="${4:-enp1s0f1np1}"
  printf '3: %s: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 state UP\n' "$iface"
  exit 0
fi
exit 0
EOF

  cat >"$tmp/bin/sudo" <<'EOF'
#!/usr/bin/env bash
printf 'sudo %s\n' "$*" >>"${MOCK_STATE_DIR}/commands.log"
exec "$@"
EOF

  cat >"$tmp/bin/ping" <<'EOF'
#!/usr/bin/env bash
printf 'ping %s\n' "$*" >>"${MOCK_STATE_DIR}/commands.log"
if [ "${MOCK_PING_FAIL:-0}" = "1" ]; then
  exit 1
fi
exit 0
EOF

  cat >"$tmp/bin/docker" <<'EOF'
#!/usr/bin/env bash
printf 'docker %s\n' "$*" >>"${MOCK_STATE_DIR}/commands.log"
case "${1:-}" in
  info)
    exit 0
    ;;
  image)
    if [ "${2:-}" = "inspect" ]; then
      if [ "${MOCK_MISSING_IMAGE:-0}" = "1" ] && [ ! -f "${MOCK_STATE_DIR}/image_pulled" ]; then
        exit 1
      fi
      exit 0
    fi
    ;;
  pull)
    touch "${MOCK_STATE_DIR}/image_pulled"
    exit 0
    ;;
  inspect)
    if [ "${MOCK_CONTAINER_STOPPED:-0}" = "1" ]; then
      printf 'false\n'
    else
      printf 'true\n'
    fi
    exit 0
    ;;
  logs)
    if [ "${MOCK_VLLM_TRACEBACK:-0}" = "1" ]; then
      printf 'Traceback (most recent call last):\nRuntimeError: NCCL timeout\n'
    else
      printf 'INFO: Ray head started\nINFO: Application startup complete.\n'
    fi
    exit 0
    ;;
  exec)
    shift
    while [[ "${1:-}" == -* ]]; do
      shift
    done
    container="$1"
    shift
    if [ "$*" = "ray status" ] || printf '%s' "$*" | grep -Fq "ray status"; then
      printf '2.0/2.0 GPU\nnode_head\nnode_worker\n'
      exit 0
    fi
    if printf '%s' "$*" | grep -Fq "/tmp/vllm-serve.log"; then
      if [ "${MOCK_VLLM_TRACEBACK:-0}" = "1" ]; then
        printf 'Traceback (most recent call last):\nRuntimeError: Engine core initialization failed\n'
        exit 0
      fi
      printf 'INFO: Uvicorn running on http://0.0.0.0:8000\n'
      exit 0
    fi
    exit 0
    ;;
  run|rm|stop)
    if [ "${1:-}" = "run" ] && printf '%s' "$*" | grep -Fq "import llmcompressor"; then
      if [ "${MOCK_MISSING_LLMCOMPRESSOR:-0}" = "1" ] && [ ! -f "${MOCK_STATE_DIR}/wheels_installed" ]; then
        exit 1
      fi
      exit 0
    fi
    if [ "${1:-}" = "run" ] && printf '%s' "$*" | grep -Fq "pip download"; then
      touch "${MOCK_STATE_DIR}/wheels_installed"
      exit 0
    fi
    if [ "${1:-}" = "run" ] && printf '%s' "$*" | grep -Fq "APERTUX_RUN_QUANTIZER=1"; then
      for arg in "$@"; do
        case "$arg" in
          *:/quant_out)
            host_qdir="${arg%:/quant_out}"
            mkdir -p "$host_qdir"
            touch "$host_qdir/model-00001-of-00008.safetensors"
            printf '{"source_model":"swiss-ai/Apertus-v1.5-70B","scheme":"NVFP4"}\n' >"$host_qdir/apertux_quant_meta.json"
            ;;
        esac
      done
      exit 0
    fi
    exit 0
    ;;
esac
exit 0
EOF

  cat >"$tmp/bin/ssh" <<'EOF'
#!/usr/bin/env bash
printf 'ssh %s\n' "$*" >>"${MOCK_STATE_DIR}/commands.log"
stdin_payload="$(cat)"
printf '%s\n---\n' "$stdin_payload" >>"${MOCK_STATE_DIR}/ssh_stdin.log"
if printf '%s' "$stdin_payload" | grep -Fq "ROLE=detect_fabric"; then
  if [ "${MOCK_WORKER_NO_IP:-0}" = "1" ] && [ ! -f "${MOCK_STATE_DIR}/worker_ip_assigned" ]; then
    printf 'IFACE=%s\nIP=\n' "${MOCK_WORKER_IFACE:-enp1s0f1np1}"
    exit 0
  fi
  printf 'IFACE=%s\nIP=%s\n' "${MOCK_WORKER_IFACE:-enp1s0f1np1}" "${MOCK_WORKER_IP:-192.168.100.2}"
  exit 0
fi
if printf '%s' "$stdin_payload" | grep -Fq "ROLE=assign_ip"; then
  touch "${MOCK_STATE_DIR}/worker_ip_assigned"
  exit 0
fi
if printf '%s' "$stdin_payload" | grep -Fq "ROLE=check_cache"; then
  if [ "${MOCK_WORKER_MISSING_WEIGHTS:-0}" = "1" ] && [ ! -f "${MOCK_STATE_DIR}/worker_downloaded" ]; then
    printf 'IMAGE_OK=1\nWEIGHTS_OK=0\nDRAFT_OK=1\n'
  elif [ "${MOCK_WORKER_MISSING_DRAFT:-0}" = "1" ] && [ ! -f "${MOCK_STATE_DIR}/worker_downloaded" ]; then
    printf 'IMAGE_OK=1\nWEIGHTS_OK=1\nDRAFT_OK=0\n'
  else
    printf 'IMAGE_OK=1\nWEIGHTS_OK=1\nDRAFT_OK=1\n'
  fi
  exit 0
fi
if printf '%s' "$stdin_payload" | grep -Fq "ROLE=bootstrap"; then
  touch "${MOCK_STATE_DIR}/worker_downloaded"
  exit 0
fi
if printf '%s' "$stdin_payload" | grep -Fq "ROLE=status_worker"; then
  if [ "${MOCK_WORKER_STOPPED:-0}" = "1" ]; then
    printf 'false\n'
  else
    printf 'true\n'
  fi
  exit 0
fi
if printf '%s' "$stdin_payload" | grep -Fq "ROLE=check_quantized"; then
  if [ -f "${MOCK_STATE_DIR}/worker_quant_synced" ]; then
    printf 'QUANT_OK=1\n'
  else
    printf 'QUANT_OK=0\n'
  fi
  exit 0
fi
if printf '%s' "$*" | grep -Fq "ROLE=sync_quantized"; then
  cat >/dev/null
  touch "${MOCK_STATE_DIR}/worker_quant_synced"
  exit 0
fi
exit 0
EOF

  cat >"$tmp/bin/curl" <<'EOF'
#!/usr/bin/env bash
printf 'curl %s\n' "$*" >>"${MOCK_STATE_DIR}/commands.log"
if printf '%s' "$*" | grep -Fq "/health"; then
  if [ "${MOCK_HEALTH_FAIL:-0}" = "1" ]; then
    exit 22
  fi
  printf 'OK'
  exit 0
fi
if printf '%s' "$*" | grep -Fq "/v1/chat/completions"; then
  printf '{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"Apertus 70B ready on 2x DGX Spark."}}],"usage":{"prompt_tokens":24,"completion_tokens":128,"total_tokens":152}}\n'
  exit 0
fi
exit 0
EOF

  chmod +x "$tmp/bin/"*
  printf '%s\n' "$tmp"
}

# --- Task 1 Tests: CLI parser, defaults, --help, validation ---

test_help_and_defaults() {
  local sb out rc
  sb="$(make_sandbox)"
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" --help 2>&1)" || rc=$?
  assert_eq "0" "${rc:-0}" "help exits 0"
  assert_contains "$out" "Usage:" "help prints usage header"
  assert_contains "$out" "swiss-ai/Apertus-v1.5-70B" "help documents default 70B model"
  assert_contains "$out" "--worker" "help documents --worker flag"

  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" up --worker nvidia@spark2 --dry-run 2>&1)" || rc=$?
  assert_eq "0" "${rc:-0}" "dry-run exits 0"
  assert_contains "$out" "MODEL=swiss-ai/Apertus-v1.5-70B" "default model is Apertus-v1.5-70B"
  assert_contains "$out" "VLLM_IMAGE=ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64" "default image is arm64 v1.5 release"
  assert_contains "$out" "MAX_MODEL_LEN=262144" "default max_model_len is full 262144 context"
  assert_contains "$out" "KV_CACHE_DTYPE=fp8" "default kv_cache_dtype is fp8 for 262k context fit"
  assert_contains "$out" "SERVE_MODE=tools" "default serve mode is tools"

  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" --worker nvidia@spark2 --model swiss-ai/Apertus-70B-Instruct-2509 --mode plain --max-model-len 65536 --kv-cache-dtype auto --port 8080 --dry-run 2>&1)" || rc=$?
  assert_eq "0" "${rc:-0}" "flag overrides exit 0"
  assert_contains "$out" "MODEL=swiss-ai/Apertus-70B-Instruct-2509" "model flag override works"
  assert_contains "$out" "SERVE_MODE=plain" "mode flag override works"
  assert_contains "$out" "MAX_MODEL_LEN=65536" "max-model-len override works"
  assert_contains "$out" "KV_CACHE_DTYPE=auto" "kv-cache-dtype override works"
  assert_contains "$out" "API_PORT=8080" "port override works"

  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" bash "$SCRIPT" --bogus-flag 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "unknown flag rejected with non-zero exit"
  else
    fail "unknown flag rejected with non-zero exit" "got exit 0"
  fi

  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" bash "$SCRIPT" --worker nvidia@spark2 --mode invalidmode --dry-run 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "invalid --mode rejected with non-zero exit"
  else
    fail "invalid --mode rejected with non-zero exit" "got exit 0"
  fi

  rm -rf "$sb"
}

test_fabric_discovery_and_preflight() {
  local sb out rc
  sb="$(make_sandbox)"

  # 1. ibdev2netdev discovery across head + SSH worker
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" up --worker nvidia@spark2 --dry-run 2>&1)" || rc=$?
  assert_eq "0" "${rc:-0}" "fabric discovery succeeds in dry-run"
  assert_contains "$out" "HEAD_FABRIC_IFACE=enp1s0f1np1" "head ConnectX-7 interface detected"
  assert_contains "$out" "HEAD_FABRIC_IP=192.168.100.1" "head 200GbE IPv4 detected"
  assert_contains "$out" "WORKER_FABRIC_IFACE=enp1s0f1np1" "worker ConnectX-7 interface detected over SSH"
  assert_contains "$out" "WORKER_FABRIC_IP=192.168.100.2" "worker 200GbE IPv4 detected over SSH"

  # 2. Fallback to ip link scan when ibdev2netdev is absent
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_NO_IBDEV=1 HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" up --worker nvidia@spark2 --dry-run 2>&1)" || rc=$?
  assert_eq "0" "$rc" "fallback ip link scan succeeds without ibdev2netdev"
  assert_contains "$out" "HEAD_FABRIC_IFACE=enp1s0f0np0" "fallback selects first LOWER_UP Spark CX7 interface"

  # 3. Non-aarch64 rejected unless --allow-arch-mismatch
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_ARCH=x86_64 HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" up --worker nvidia@spark2 --dry-run 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "x86_64 rejected by default"
  else
    fail "x86_64 rejected by default" "got exit 0"
  fi
  assert_contains "$out" "aarch64" "arch error mentions aarch64"

  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_ARCH=x86_64 HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" up --worker nvidia@spark2 --allow-arch-mismatch --dry-run 2>&1)" || rc=$?
  assert_eq "0" "$rc" "--allow-arch-mismatch bypasses arch guard"

  # 4. Missing IPv4 fails without --assign-fabric-ips, succeeds with --assign-fabric-ips
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_HEAD_NO_IP=1 HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" up --worker nvidia@spark2 --dry-run 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "missing 200GbE IPv4 fails cleanly"
  else
    fail "missing 200GbE IPv4 fails cleanly" "got exit 0"
  fi
  assert_contains "$out" "ip addr add" "missing IP error gives ip addr add hint"

  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_HEAD_NO_IP=1 MOCK_WORKER_NO_IP=1 HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" up --worker nvidia@spark2 --assign-fabric-ips 2>&1)" || rc=$?
  assert_eq "0" "$rc" "--assign-fabric-ips assigns head and worker IPs"
  assert_contains "$(cat "$sb/state/commands.log")" "ip addr add 192.168.100.1/24 dev enp1s0f1np1" "head IP assigned via ip addr add"
  assert_contains "$(cat "$sb/state/ssh_stdin.log")" "ROLE=assign_ip" "worker IP assigned over SSH"

  # 5. Ping failure over fabric caught
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_PING_FAIL=1 HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" up --worker nvidia@spark2 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "200GbE ping failure aborts startup"
  else
    fail "200GbE ping failure aborts startup" "got exit 0"
  fi

  rm -rf "$sb"
}

test_local_only_bootstrap_gate() {
  local sb out rc
  sb="$(make_sandbox)"

  # 1. Cached run never calls docker pull
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" up --worker nvidia@spark2 2>&1)" || rc=$?
  assert_eq "0" "${rc:-0}" "cached run succeeds without network pull"
  assert_not_contains "$(cat "$sb/state/commands.log")" "docker pull" "cached run skips docker pull"

  # 2. Missing image without --approve-download aborts before pulling
  rm -f "$sb/state/commands.log" "$sb/state/image_pulled"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_MISSING_IMAGE=1 HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" up --worker nvidia@spark2 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "missing image aborts without --approve-download"
  else
    fail "missing image aborts without --approve-download" "got exit 0"
  fi
  assert_contains "$out" "--approve-download" "missing artifact error explains --approve-download"
  assert_not_contains "$(cat "$sb/state/commands.log")" "docker pull" "unapproved run never executes docker pull"

  # 3. Missing worker weights without --approve-download aborts
  rm -f "$sb/state/commands.log" "$sb/state/ssh_stdin.log"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_WORKER_MISSING_WEIGHTS=1 HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" up --worker nvidia@spark2 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "missing worker weights abort without --approve-download"
  else
    fail "missing worker weights abort without --approve-download" "got exit 0"
  fi
  assert_not_contains "$(cat "$sb/state/ssh_stdin.log")" "ROLE=bootstrap" "unapproved run never bootstraps worker"

  # 4. With --approve-download, pulls missing image and bootstraps worker without leaking HF_TOKEN
  rm -f "$sb/state/commands.log" "$sb/state/ssh_stdin.log"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_MISSING_IMAGE=1 MOCK_WORKER_MISSING_WEIGHTS=1 HF_TOKEN="hf_secret_do_not_leak" HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" up --worker nvidia@spark2 --approve-download 2>&1)" || rc=$?
  assert_eq "0" "$rc" "approved bootstrap succeeds"
  assert_contains "$(cat "$sb/state/commands.log")" "docker pull ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64" "approved bootstrap pulls image"
  assert_contains "$(cat "$sb/state/ssh_stdin.log")" "ROLE=bootstrap" "approved bootstrap downloads worker weights"
  assert_not_contains "$out" "hf_secret_do_not_leak" "HF_TOKEN never leaked to output"

  rm -rf "$sb"
}

test_cluster_orchestration_and_modes() {
  local sb out rc cmds ssh_in
  sb="$(make_sandbox)"

  # 1. Full cluster up in default tools mode with 262k + fp8 KV cache + prefix caching
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" up --worker nvidia@spark2 2>&1)" || rc=$?
  assert_eq "0" "$rc" "cluster up exits 0"
  cmds="$(cat "$sb/state/commands.log")"
  ssh_in="$(cat "$sb/state/ssh_stdin.log")"

  assert_contains "$cmds" "VLLM_HOST_IP=192.168.100.1" "head sets VLLM_HOST_IP to 200GbE IP"
  assert_contains "$cmds" "NCCL_SOCKET_IFNAME=enp1s0f1np1" "head pins NCCL_SOCKET_IFNAME"
  assert_contains "$cmds" "GLOO_SOCKET_IFNAME=enp1s0f1np1" "head pins GLOO_SOCKET_IFNAME"
  assert_contains "$cmds" "TP_SOCKET_IFNAME=enp1s0f1np1" "head pins TP_SOCKET_IFNAME"
  assert_contains "$cmds" "UCX_NET_DEVICES=enp1s0f1np1" "head pins UCX_NET_DEVICES"
  assert_contains "$cmds" "HF_HUB_OFFLINE=1" "head enforces HF_HUB_OFFLINE=1"
  assert_contains "$cmds" "VLLM_NO_USAGE_STATS=1" "head disables vLLM telemetry"
  assert_contains "$cmds" "ray start --head --node-ip-address=192.168.100.1 --port=6379 --disable-usage-stats --block" "head starts Ray head on 200GbE IP"
  assert_contains "$ssh_in" "ROLE=start_worker" "worker started over SSH inline payload"
  assert_contains "$cmds" "192.168.100.1 6379 192.168.100.2 enp1s0f1np1" "worker SSH receives head IP, Ray port, worker IP, and CX7 iface"
  assert_contains "$ssh_in" 'ray start --address=$head_ip:$rport --node-ip-address=$wip --disable-usage-stats --block' "worker joins head 200GbE Ray port"

  assert_contains "$cmds" "vllm serve swiss-ai/Apertus-v1.5-70B" "vllm serve targets Apertus-v1.5-70B"
  assert_contains "$cmds" "--tensor-parallel-size 2" "vllm serve sets TP=2"
  assert_contains "$cmds" "--distributed-executor-backend ray" "vllm serve uses ray executor"
  assert_contains "$cmds" "--kv-cache-dtype fp8" "vllm serve uses fp8 KV cache"
  assert_contains "$cmds" "--enable-prefix-caching" "vllm serve enables prefix caching"
  assert_contains "$cmds" "--max-model-len 262144" "vllm serve sets 262k context"
  assert_contains "$cmds" "--compilation-config.pass_config.fuse_allreduce_rms false" "vllm serve disables fuse_allreduce_rms for TP>1 stability"
  assert_contains "$cmds" "--enable-auto-tool-choice --tool-call-parser apertus" "tools mode enables apertus tool parser"

  # 2. Thinking mode flags
  rm -f "$sb/state/commands.log" "$sb/state/ssh_stdin.log"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" up --worker nvidia@spark2 --mode thinking 2>&1)" || rc=$?
  assert_eq "0" "$rc" "thinking mode cluster up exits 0"
  cmds="$(cat "$sb/state/commands.log")"
  assert_contains "$cmds" "--reasoning-parser apertus --default-chat-template-kwargs.enable_thinking true" "thinking mode configures reasoning parser"
  assert_not_contains "$cmds" "--tool-call-parser" "thinking mode omits tool-call-parser"

  # 3. Plain mode + --no-prefix-caching + --nccl-ib-disable
  rm -f "$sb/state/commands.log" "$sb/state/ssh_stdin.log"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" up --worker nvidia@spark2 --mode plain --no-prefix-caching --nccl-ib-disable 2>&1)" || rc=$?
  assert_eq "0" "$rc" "plain mode cluster up exits 0"
  cmds="$(cat "$sb/state/commands.log")"
  assert_contains "$cmds" "NCCL_IB_DISABLE=1" "--nccl-ib-disable sets NCCL_IB_DISABLE=1"
  assert_not_contains "$cmds" "--enable-prefix-caching" "--no-prefix-caching omits prefix caching flag"
  assert_not_contains "$cmds" "--tool-call-parser" "plain mode omits tool parser"
  assert_not_contains "$cmds" "--reasoning-parser" "plain mode omits reasoning parser"

  # 4. Standalone --worker-only mode
  rm -f "$sb/state/commands.log" "$sb/state/ssh_stdin.log"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" --worker-only 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "--worker-only without --head-ip fails"
  else
    fail "--worker-only without --head-ip fails" "got exit 0"
  fi

  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_HEAD_IP=192.168.100.2 HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" --worker-only --head-ip 192.168.100.1 2>&1)" || rc=$?
  assert_eq "0" "$rc" "--worker-only with --head-ip succeeds"
  cmds="$(cat "$sb/state/commands.log")"
  assert_contains "$cmds" "ray start --address=192.168.100.1:6379 --node-ip-address=192.168.100.2 --disable-usage-stats --block" "standalone worker joins head Ray"
  assert_not_contains "$cmds" "ssh " "standalone worker makes zero SSH calls"

  # 5. Teardown (down) across both nodes
  rm -f "$sb/state/commands.log" "$sb/state/ssh_stdin.log"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" bash "$SCRIPT" down --worker nvidia@spark2 2>&1)" || rc=$?
  assert_eq "0" "$rc" "down --worker exits 0"
  assert_contains "$(cat "$sb/state/commands.log")" "docker rm -f apertux-node" "down removes local apertux-node"
  assert_contains "$(cat "$sb/state/ssh_stdin.log")" "ROLE=stop_worker" "down removes remote apertux-node over SSH"

  rm -rf "$sb"
}

test_liveness_smoke_status_and_pipe() {
  local sb out rc cmds
  sb="$(make_sandbox)"

  # 1. Traceback in /tmp/vllm-serve.log caught immediately during startup
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_HEALTH_FAIL=1 MOCK_VLLM_TRACEBACK=1 HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 APERTUX_POLL_INTERVAL=0 APERTUX_HEALTH_WAIT_TRIES=3 bash "$SCRIPT" up --worker nvidia@spark2 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "startup traceback in vllm log aborts non-zero"
  else
    fail "startup traceback in vllm log aborts non-zero" "got exit 0"
  fi
  assert_contains "$out" "Traceback" "startup failure surfaces recent vllm log lines"

  # 2. Standalone smoke subcommand hits /v1/chat/completions
  rm -f "$sb/state/commands.log"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" bash "$SCRIPT" smoke --endpoint http://127.0.0.1:8000 2>&1)" || rc=$?
  assert_eq "0" "$rc" "smoke subcommand exits 0"
  assert_contains "$(cat "$sb/state/commands.log")" "http://127.0.0.1:8000/v1/chat/completions" "smoke calls /v1/chat/completions"
  assert_contains "$out" "Apertus 70B ready on 2x DGX Spark." "smoke prints model completion"

  # 3. status subcommand checks container, Ray GPUs, and /health
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" bash "$SCRIPT" status --worker nvidia@spark2 2>&1)" || rc=$?
  assert_eq "0" "$rc" "status exits 0 when healthy"
  assert_contains "$out" "2.0/2.0 GPU" "status reports Ray GPU count"

  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_CONTAINER_STOPPED=1 bash "$SCRIPT" status 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "status exits non-zero when container is stopped"
  else
    fail "status exits non-zero when container is stopped" "got exit 0"
  fi

  # 4. logs subcommand surfaces /tmp/vllm-serve.log
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" bash "$SCRIPT" logs 2>&1)" || rc=$?
  assert_eq "0" "$rc" "logs subcommand exits 0"
  assert_contains "$out" "Uvicorn running" "logs reads vllm-serve.log"

  # 5. End-to-end piped curl | bash simulation ($0 == bash, stdin consumed by script)
  rm -f "$sb/state/commands.log" "$sb/state/ssh_stdin.log"
  rc=0
  out="$(cat "$SCRIPT" | PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash -s -- --worker nvidia@spark2 2>&1)" || rc=$?
  assert_eq "0" "$rc" "piped curl | bash -s execution succeeds end-to-end"
  cmds="$(cat "$sb/state/commands.log")"
  assert_contains "$cmds" "/health" "up polls /health before declaring ready"
  assert_contains "$cmds" "/v1/chat/completions" "up executes live curl chat completion check"
  assert_contains "$out" "curl -sS http://" "up prints ready-to-use curl command for clients"

  rm -rf "$sb"
}

test_logo_assets() {
  local asset
  for asset in assets/logo.svg assets/logo-light.svg assets/logo-dark.svg assets/icon.svg assets/icon.png; do
    if [ -f "$ROOT_DIR/$asset" ]; then
      pass "$asset exists"
    else
      fail "$asset exists" "missing $ROOT_DIR/$asset"
    fi
  done

  if python3 - "$ROOT_DIR" <<'PY'
import pathlib, struct, sys, xml.etree.ElementTree as ET
root = pathlib.Path(sys.argv[1])
ns = {"svg": "http://www.w3.org/2000/svg"}

for name in ["logo.svg", "logo-light.svg", "logo-dark.svg", "icon.svg"]:
    p = root / "assets" / name
    if not p.is_file():
        sys.exit(1)
    text = p.read_text()
    el = ET.fromstring(text)
    assert el.attrib.get("role") == "img", f"{name} missing role=img"
    title = el.find("svg:title", ns)
    assert title is not None and title.text and "apertux" in title.text.lower(), f"{name} missing accessible <title>"

for adaptive in ["logo.svg", "icon.svg"]:
    text = (root / "assets" / adaptive).read_text()
    assert "--apertux-ink" in text and "--apertux-accent" in text, f"{adaptive} missing CSS tokens"
    assert "prefers-color-scheme: dark" in text, f"{adaptive} missing dark token swap"

png = (root / "assets" / "icon.png").read_bytes()
assert png[:8] == b"\x89PNG\r\n\x1a\n", "invalid PNG header"
w, h = struct.unpack(">II", png[16:24])
assert (w, h) == (512, 512), f"expected 512x512 icon.png, got {w}x{h}"
PY
  then
    pass "SVG XML structure, a11y titles, CSS dark-mode tokens, and 512x512 PNG verified"
  else
    fail "SVG XML structure, a11y titles, CSS dark-mode tokens, and 512x512 PNG verified" "validation failed"
  fi

  local readme
  readme="$(cat "$ROOT_DIR/README.md")"
  assert_contains "$readme" "assets/logo-dark.svg" "README references dark logo variant"
  assert_contains "$readme" "assets/logo-light.svg" "README references light logo variant"
  assert_contains "$readme" "## References" "README includes References section"
  assert_contains "$readme" "https://apertus-ai.org" "README references Apertus project site"
  assert_contains "$readme" "Oleg Lavrovsky" "README credits Oleg Lavrovsky"
}

test_turbo_specdec_and_draft_gate() {
  local sb out rc cmds
  sb="$(make_sandbox)"

  # 1. --turbo preset dry-run config + speculative-config draft_model JSON + --dtype auto
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" QUANT_BASE_DIR="$sb/quant" bash "$SCRIPT" up --worker nvidia@spark2 --turbo --dry-run 2>&1)" || rc=$?
  assert_eq "0" "$rc" "--turbo dry-run exits 0"
  assert_contains "$out" "QUANT_SCHEME=nvfp4" "--turbo defaults QUANT_SCHEME=nvfp4"
  assert_contains "$out" "SPEC_MODE=draft" "--turbo defaults SPEC_MODE=draft"
  assert_contains "$out" "DRAFT_MODEL=swiss-ai/Apertus-v1.1-1.5B-Instruct" "--turbo defaults draft model to Apertus-v1.1-1.5B-Instruct"
  assert_contains "$out" "--dtype auto" "quantized serve uses --dtype auto for compressed-tensors"
  assert_contains "$out" '--speculative-config {"method":"draft_model","model":"swiss-ai/Apertus-v1.1-1.5B-Instruct","num_speculative_tokens":5}' "--turbo builds draft_model speculative-config"

  # 2. --spec-mode ngram builds ngram speculative-config without draft model
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" bash "$SCRIPT" up --worker nvidia@spark2 --quant fp8 --spec-mode ngram --spec-tokens 6 --prompt-lookup-max 4 --dry-run 2>&1)" || rc=$?
  assert_eq "0" "$rc" "--spec-mode ngram dry-run exits 0"
  assert_contains "$out" "QUANT_SCHEME=fp8" "--quant fp8 sets QUANT_SCHEME=fp8"
  assert_contains "$out" '--speculative-config {"method":"ngram","num_speculative_tokens":6,"prompt_lookup_max":4}' "ngram builds lookup speculative-config"

  # 3. Draft model local-only cache gate: rejects missing draft without --approve-download, downloads when approved
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" up --worker nvidia@spark2 --spec-mode draft 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "missing local draft model aborts without --approve-download"
  else
    fail "missing local draft model aborts without --approve-download" "got exit 0"
  fi
  assert_contains "$out" "Apertus-v1.1-1.5B-Instruct" "missing draft error names draft checkpoint"

  # Populate mock local draft cache after docker snapshot_download or test --approve-download
  mkdir -p "$sb/hf/hub/models--swiss-ai--Apertus-v1.1-1.5B-Instruct/snapshots/def456"
  touch "$sb/hf/hub/models--swiss-ai--Apertus-v1.1-1.5B-Instruct/snapshots/def456/model.safetensors"
  rm -f "$sb/state/commands.log" "$sb/state/ssh_stdin.log" "$sb/state/worker_downloaded"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_WORKER_MISSING_DRAFT=1 HF_CACHE_DIR="$sb/hf" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" up --worker nvidia@spark2 --spec-mode draft --approve-download 2>&1)" || rc=$?
  assert_eq "0" "$rc" "draft mode succeeds and bootstraps missing worker draft when approved"
  assert_contains "$(cat "$sb/state/ssh_stdin.log")" "ROLE=bootstrap" "worker bootstrap triggered for missing draft model"

  rm -rf "$sb"
}

test_quantize_sync_bench_and_piped_turbo() {
  local sb out rc cmds ssh_in
  sb="$(make_sandbox)"
  mkdir -p "$sb/hf/hub/models--swiss-ai--Apertus-v1.1-1.5B-Instruct/snapshots/def456"
  touch "$sb/hf/hub/models--swiss-ai--Apertus-v1.1-1.5B-Instruct/snapshots/def456/model.safetensors"

  # 1. Missing llmcompressor in container requires --approve-download under local-only rule
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" MOCK_MISSING_LLMCOMPRESSOR=1 HF_CACHE_DIR="$sb/hf" QUANT_BASE_DIR="$sb/quant" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" quantize --quant nvfp4 --worker nvidia@spark2 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    pass "missing llmcompressor wheel aborts without --approve-download"
  else
    fail "missing llmcompressor wheel aborts without --approve-download" "got exit 0"
  fi

  # 2. quantize subcommand creates local NVFP4 checkpoint and syncs to Spark 2
  rm -f "$sb/state/commands.log" "$sb/state/ssh_stdin.log"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" QUANT_BASE_DIR="$sb/quant" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" quantize --quant nvfp4 --worker nvidia@spark2 2>&1)" || rc=$?
  assert_eq "0" "$rc" "quantize --quant nvfp4 succeeds"
  cmds="$(cat "$sb/state/commands.log")"
  assert_contains "$cmds" "APERTUX_RUN_QUANTIZER=1" "quantize launches container quantizer"
  assert_contains "$cmds" "ROLE=sync_quantized" "quantize syncs compressed checkpoint to Spark 2"

  # 3. Idempotent second quantize skips container quantizer when both Sparks have checkpoint
  rm -f "$sb/state/commands.log"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" QUANT_BASE_DIR="$sb/quant" APERTUX_DISABLE_TTY_PROMPT=1 bash "$SCRIPT" quantize --quant nvfp4 --worker nvidia@spark2 2>&1)" || rc=$?
  assert_eq "0" "$rc" "second quantize exits 0 idempotently"
  assert_not_contains "$(cat "$sb/state/commands.log")" "APERTUX_RUN_QUANTIZER=1" "idempotent quantize skips re-running llmcompressor"

  # 4. bench subcommand reports completion_tokens and tok/s
  rm -f "$sb/state/commands.log"
  rc=0
  out="$(PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" bash "$SCRIPT" bench --endpoint http://127.0.0.1:8000 --max-tokens 128 2>&1)" || rc=$?
  assert_eq "0" "$rc" "bench subcommand exits 0"
  assert_contains "$out" "128 tokens" "bench reports generated completion_tokens"
  assert_contains "$out" "tok/s" "bench reports decode throughput in tok/s"

  # 5. End-to-end piped curl | bash --turbo (fresh quant dir, $0 == bash, embedded quantizer + worker /models/target mount)
  rm -rf "$sb/quant" "$sb/state/worker_quant_synced" "$sb/state/commands.log" "$sb/state/ssh_stdin.log"
  rc=0
  out="$(cat "$SCRIPT" | PATH="$sb/bin:$PATH" MOCK_STATE_DIR="$sb/state" HF_CACHE_DIR="$sb/hf" QUANT_BASE_DIR="$sb/quant" APERTUX_DISABLE_TTY_PROMPT=1 bash -s -- --worker nvidia@spark2 --turbo 2>&1)" || rc=$?
  assert_eq "0" "$rc" "piped curl | bash --turbo succeeds end-to-end"
  cmds="$(cat "$sb/state/commands.log")"
  ssh_in="$(cat "$sb/state/ssh_stdin.log")"
  assert_contains "$cmds" "APERTUX_RUN_QUANTIZER=1" "piped --turbo runs embedded quantizer when checkpoint absent"
  assert_contains "$cmds" "ROLE=sync_quantized" "piped --turbo syncs quantized checkpoint to Spark 2"
  assert_contains "$cmds" ":/models/target:ro" "head mounts quantized checkpoint at /models/target"
  assert_contains "$ssh_in" ":/models/target:ro" "worker mounts quantized checkpoint at /models/target"
  assert_contains "$cmds" '--speculative-config {"method":"draft_model","model":"swiss-ai/Apertus-v1.1-1.5B-Instruct","num_speculative_tokens":5}' "piped --turbo passes draft speculative-config"

  rm -rf "$sb"
}

test_help_and_defaults
test_fabric_discovery_and_preflight
test_local_only_bootstrap_gate
test_cluster_orchestration_and_modes
test_liveness_smoke_status_and_pipe
test_logo_assets
test_turbo_specdec_and_draft_gate
test_quantize_sync_bench_and_piped_turbo

printf '\nSummary: %d passed, %d failed\n' "$PASS_COUNT" "$FAIL_COUNT"
if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi
