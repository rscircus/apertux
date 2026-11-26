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
    printf 'IMAGE_OK=1\nWEIGHTS_OK=0\n'
  else
    printf 'IMAGE_OK=1\nWEIGHTS_OK=1\n'
  fi
  exit 0
fi
if printf '%s' "$stdin_payload" | grep -Fq "ROLE=bootstrap"; then
  touch "${MOCK_STATE_DIR}/worker_downloaded"
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
  printf '{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"Apertus 70B ready on 2x DGX Spark."}}]}\n'
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

test_help_and_defaults

printf '\nSummary: %d passed, %d failed\n' "$PASS_COUNT" "$FAIL_COUNT"
if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi
