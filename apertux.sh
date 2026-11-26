#!/usr/bin/env bash
set -euo pipefail

# apertux.sh - Deploy Swiss-AI Apertus across two NVIDIA DGX Sparks (TP=2)

ACTION="up"
ROLE_MODE="cluster" # cluster | head | worker-only
WORKER_SSH="${WORKER_SSH:-}"
HEAD_IP="${HEAD_IP:-}"
MODEL="${MODEL:-swiss-ai/Apertus-v1.5-70B}"
VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64}"
CONTAINER_NAME="${CONTAINER_NAME:-apertux-node}"
FABRIC_IFACE="${FABRIC_IFACE:-}"
WORKER_FABRIC_IFACE="${WORKER_FABRIC_IFACE:-}"
ASSIGN_FABRIC_IPS="${ASSIGN_FABRIC_IPS:-0}"
HEAD_ASSIGN_CIDR="${HEAD_ASSIGN_CIDR:-192.168.100.1/24}"
WORKER_ASSIGN_CIDR="${WORKER_ASSIGN_CIDR:-192.168.100.2/24}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
SERVE_MODE="${SERVE_MODE:-tools}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
APPROVE_DOWNLOAD="${APPROVE_DOWNLOAD:-0}"
ALLOW_ARCH_MISMATCH="${ALLOW_ARCH_MISMATCH:-0}"
DRY_RUN=0
FOLLOW_LOGS=0
SMOKE_ENDPOINT=""

usage() {
  cat <<'EOF'
Usage:
  apertux.sh [up] --worker <user@spark2> [options]
  apertux.sh --head [options]
  apertux.sh --worker-only --head-ip <200g-head-ip> [options]
  apertux.sh status [--worker <user@spark2>]
  apertux.sh smoke [--endpoint http://127.0.0.1:8000]
  apertux.sh logs [--follow]
  apertux.sh down [--worker <user@spark2>]

Defaults:
  MODEL=swiss-ai/Apertus-v1.5-70B
  VLLM_IMAGE=ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64
  MAX_MODEL_LEN=262144
  KV_CACHE_DTYPE=fp8
  ENABLE_PREFIX_CACHING=1
  GPU_MEM_UTIL=0.88
  SERVE_MODE=tools (tools | thinking | plain)
  API_PORT=8000

Options:
  --worker <user@host>        SSH target for Spark 2
  --head                      Start head node without SSH worker dispatch
  --worker-only               Start Ray worker container only (requires --head-ip)
  --head-ip <ipv4>            Spark 1 200GbE IPv4 for --worker-only mode
  --model <hf-repo>           Model repo or local snapshot path
  --image <docker-image>      vLLM container image
  --fabric-iface <ifname>     Override 200GbE ConnectX-7 interface on Spark 1
  --worker-fabric-iface <if>  Override 200GbE ConnectX-7 interface on Spark 2
  --assign-fabric-ips         Assign 192.168.100.1/24 and .2/24 if 200GbE lacks IPv4
  --max-model-len <tokens>    Max context length (default: 262144)
  --kv-cache-dtype <dtype>    KV cache dtype: fp8 | auto (default: fp8)
  --no-prefix-caching         Disable vLLM automatic prefix caching
  --gpu-mem-util <ratio>      vLLM GPU memory utilization (default: 0.88)
  --mode <mode>               Parser profile: tools | thinking | plain
  --port <port>               OpenAI HTTP server port on Spark 1 (default: 8000)
  --hf-cache <dir>            Host Hugging Face cache directory
  --nccl-ib-disable           Set NCCL_IB_DISABLE=1 (TCP over 200GbE fallback)
  --approve-download          Authorize first-time image pull and weight download
  --allow-arch-mismatch       Bypass aarch64 host check
  --endpoint <url>            Base URL for smoke check
  --follow, -f                Follow container logs in logs subcommand
  --dry-run                   Print resolved configuration and commands only
  --help, -h                  Show this help text
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

parse_args() {
  if [ "$#" -gt 0 ]; then
    case "$1" in
      up|down|status|smoke|logs)
        ACTION="$1"
        shift
        ;;
    esac
  fi

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --worker)
        [ "$#" -ge 2 ] || die "--worker requires a value"
        WORKER_SSH="$2"
        shift 2
        ;;
      --head)
        ROLE_MODE="head"
        shift
        ;;
      --worker-only)
        ROLE_MODE="worker-only"
        shift
        ;;
      --head-ip)
        [ "$#" -ge 2 ] || die "--head-ip requires a value"
        HEAD_IP="$2"
        shift 2
        ;;
      --model)
        [ "$#" -ge 2 ] || die "--model requires a value"
        MODEL="$2"
        shift 2
        ;;
      --image)
        [ "$#" -ge 2 ] || die "--image requires a value"
        VLLM_IMAGE="$2"
        shift 2
        ;;
      --fabric-iface)
        [ "$#" -ge 2 ] || die "--fabric-iface requires a value"
        FABRIC_IFACE="$2"
        shift 2
        ;;
      --worker-fabric-iface)
        [ "$#" -ge 2 ] || die "--worker-fabric-iface requires a value"
        WORKER_FABRIC_IFACE="$2"
        shift 2
        ;;
      --assign-fabric-ips)
        ASSIGN_FABRIC_IPS=1
        shift
        ;;
      --max-model-len)
        [ "$#" -ge 2 ] || die "--max-model-len requires a value"
        MAX_MODEL_LEN="$2"
        shift 2
        ;;
      --kv-cache-dtype)
        [ "$#" -ge 2 ] || die "--kv-cache-dtype requires a value"
        KV_CACHE_DTYPE="$2"
        shift 2
        ;;
      --no-prefix-caching)
        ENABLE_PREFIX_CACHING=0
        shift
        ;;
      --gpu-mem-util)
        [ "$#" -ge 2 ] || die "--gpu-mem-util requires a value"
        GPU_MEM_UTIL="$2"
        shift 2
        ;;
      --mode)
        [ "$#" -ge 2 ] || die "--mode requires a value"
        SERVE_MODE="$2"
        shift 2
        ;;
      --port)
        [ "$#" -ge 2 ] || die "--port requires a value"
        API_PORT="$2"
        shift 2
        ;;
      --hf-cache)
        [ "$#" -ge 2 ] || die "--hf-cache requires a value"
        HF_CACHE_DIR="$2"
        shift 2
        ;;
      --nccl-ib-disable)
        NCCL_IB_DISABLE=1
        shift
        ;;
      --approve-download)
        APPROVE_DOWNLOAD=1
        shift
        ;;
      --allow-arch-mismatch)
        ALLOW_ARCH_MISMATCH=1
        shift
        ;;
      --endpoint)
        [ "$#" -ge 2 ] || die "--endpoint requires a value"
        SMOKE_ENDPOINT="$2"
        shift 2
        ;;
      --follow|-f)
        FOLLOW_LOGS=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1 (run with --help for usage)"
        ;;
    esac
  done

  case "$SERVE_MODE" in
    tools|thinking|plain) ;;
    *) die "Invalid --mode '$SERVE_MODE' (expected: tools, thinking, plain)" ;;
  esac

  case "$KV_CACHE_DTYPE" in
    fp8|fp8_e4m3|fp8_e5m2|auto) ;;
    *) die "Invalid --kv-cache-dtype '$KV_CACHE_DTYPE' (expected: fp8, auto)" ;;
  esac
}

HEAD_FABRIC_IFACE=""
HEAD_FABRIC_IP=""
RESOLVED_WORKER_IFACE=""
WORKER_FABRIC_IP=""

SPARK_CX7_IFACES="enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1"

check_host_preflight() {
  local arch
  arch="$(uname -m)"
  if [ "$arch" != "aarch64" ] && [ "$ALLOW_ARCH_MISMATCH" != "1" ]; then
    die "Expected aarch64 (DGX Spark GB10), got '$arch' (use --allow-arch-mismatch to override)"
  fi
  command -v docker >/dev/null 2>&1 || die "docker not found on Spark 1"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found on Spark 1"
  if [ "$DRY_RUN" != "1" ]; then
    docker info >/dev/null 2>&1 || die "docker daemon is not reachable on Spark 1"
    nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed on Spark 1"
  fi
}

extract_ipv4() {
  local iface="$1"
  ip -4 -o addr show dev "$iface" 2>/dev/null | awk '{
    for (i = 1; i <= NF; i++) {
      if ($i == "inet") {
        sub(/\/.*/, "", $(i + 1))
        print $(i + 1)
        exit
      }
    }
  }'
}

find_active_cx7_iface() {
  local cand
  if command -v ibdev2netdev >/dev/null 2>&1; then
    cand="$(ibdev2netdev 2>/dev/null | awk '/\(Up\)/ {
      for (i = 1; i <= NF; i++) {
        if ($i == "==>") {
          print $(i + 1)
          exit
        }
      }
    }' || true)"
    if [ -n "$cand" ]; then
      printf '%s\n' "$cand"
      return 0
    fi
  fi

  for cand in $SPARK_CX7_IFACES; do
    if ip link show dev "$cand" 2>/dev/null | grep -Fq "LOWER_UP"; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

detect_local_fabric() {
  if [ -n "$FABRIC_IFACE" ]; then
    HEAD_FABRIC_IFACE="$FABRIC_IFACE"
  else
    HEAD_FABRIC_IFACE="$(find_active_cx7_iface || true)"
  fi
  [ -n "$HEAD_FABRIC_IFACE" ] || die "No active ConnectX-7 200GbE port found on Spark 1 (check QSFP cable or pass --fabric-iface)"

  HEAD_FABRIC_IP="$(extract_ipv4 "$HEAD_FABRIC_IFACE")"
  if [ -z "$HEAD_FABRIC_IP" ]; then
    if [ "$ASSIGN_FABRIC_IPS" = "1" ] && [ "$DRY_RUN" != "1" ]; then
      sudo ip addr add "$HEAD_ASSIGN_CIDR" dev "$HEAD_FABRIC_IFACE"
      sudo ip link set "$HEAD_FABRIC_IFACE" up
      HEAD_FABRIC_IP="$(extract_ipv4 "$HEAD_FABRIC_IFACE")"
    fi
  fi
  if [ -z "$HEAD_FABRIC_IP" ]; then
    die "Interface $HEAD_FABRIC_IFACE has no IPv4 on Spark 1. Run 'sudo ip addr add $HEAD_ASSIGN_CIDR dev $HEAD_FABRIC_IFACE' or pass --assign-fabric-ips"
  fi
}

detect_remote_fabric() {
  [ -n "$WORKER_SSH" ] || die "--worker <user@spark2> is required in cluster mode (or pass --head)"
  local out
  out="$(ssh "$WORKER_SSH" bash -s -- "$WORKER_FABRIC_IFACE" "$ALLOW_ARCH_MISMATCH" <<'REMOTE_EOF'
set -euo pipefail
# ROLE=detect_fabric
req_iface="${1:-}"
allow_arch="${2:-0}"
arch="$(uname -m)"
if [ "$arch" != "aarch64" ] && [ "$allow_arch" != "1" ]; then
  printf 'ERROR: Spark 2 arch is %s, expected aarch64\n' "$arch" >&2
  exit 1
fi
if [ -n "$req_iface" ]; then
  iface="$req_iface"
else
  iface=""
  if command -v ibdev2netdev >/dev/null 2>&1; then
    iface="$(ibdev2netdev 2>/dev/null | awk '/\(Up\)/ { for (i=1; i<=NF; i++) if ($i=="==>") { print $(i+1); exit } }' || true)"
  fi
  if [ -z "$iface" ]; then
    for c in enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1; do
      if ip link show dev "$c" 2>/dev/null | grep -Fq "LOWER_UP"; then
        iface="$c"
        break
      fi
    done
  fi
fi
[ -n "$iface" ] || { printf 'ERROR: No active ConnectX-7 port found on Spark 2\n' >&2; exit 1; }
ip4="$(ip -4 -o addr show dev "$iface" 2>/dev/null | awk '{ for (i=1; i<=NF; i++) if ($i=="inet") { sub(/\/.*/, "", $(i+1)); print $(i+1); exit } }')"
printf 'IFACE=%s\nIP=%s\n' "$iface" "$ip4"
REMOTE_EOF
)"
  RESOLVED_WORKER_IFACE="$(printf '%s\n' "$out" | awk -F= '/^IFACE=/ { print $2; exit }')"
  WORKER_FABRIC_IP="$(printf '%s\n' "$out" | awk -F= '/^IP=/ { print $2; exit }')"

  if [ -z "$WORKER_FABRIC_IP" ] && [ "$ASSIGN_FABRIC_IPS" = "1" ] && [ "$DRY_RUN" != "1" ]; then
    ssh "$WORKER_SSH" bash -s -- "$RESOLVED_WORKER_IFACE" "$WORKER_ASSIGN_CIDR" <<'REMOTE_EOF'
set -euo pipefail
# ROLE=assign_ip
sudo ip addr add "$2" dev "$1"
sudo ip link set "$1" up
REMOTE_EOF
    out="$(ssh "$WORKER_SSH" bash -s -- "$RESOLVED_WORKER_IFACE" "$ALLOW_ARCH_MISMATCH" <<'REMOTE_EOF'
set -euo pipefail
# ROLE=detect_fabric
iface="$1"
ip4="$(ip -4 -o addr show dev "$iface" 2>/dev/null | awk '{ for (i=1; i<=NF; i++) if ($i=="inet") { sub(/\/.*/, "", $(i+1)); print $(i+1); exit } }')"
printf 'IFACE=%s\nIP=%s\n' "$iface" "$ip4"
REMOTE_EOF
)"
    WORKER_FABRIC_IP="$(printf '%s\n' "$out" | awk -F= '/^IP=/ { print $2; exit }')"
  fi

  if [ -z "$WORKER_FABRIC_IP" ]; then
    die "Interface $RESOLVED_WORKER_IFACE has no IPv4 on Spark 2 ($WORKER_SSH). Run 'sudo ip addr add $WORKER_ASSIGN_CIDR dev $RESOLVED_WORKER_IFACE' or pass --assign-fabric-ips"
  fi
}

verify_fabric_ping() {
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  ping -c 1 -W 2 "$WORKER_FABRIC_IP" >/dev/null 2>&1 || die "Cannot reach Spark 2 over 200GbE fabric ($HEAD_FABRIC_IP -> $WORKER_FABRIC_IP on $HEAD_FABRIC_IFACE)"
}

print_config() {
  cat <<EOF
ACTION=$ACTION
ROLE_MODE=$ROLE_MODE
WORKER_SSH=$WORKER_SSH
MODEL=$MODEL
VLLM_IMAGE=$VLLM_IMAGE
HEAD_FABRIC_IFACE=$HEAD_FABRIC_IFACE
HEAD_FABRIC_IP=$HEAD_FABRIC_IP
WORKER_FABRIC_IFACE=$RESOLVED_WORKER_IFACE
WORKER_FABRIC_IP=$WORKER_FABRIC_IP
MAX_MODEL_LEN=$MAX_MODEL_LEN
KV_CACHE_DTYPE=$KV_CACHE_DTYPE
ENABLE_PREFIX_CACHING=$ENABLE_PREFIX_CACHING
GPU_MEM_UTIL=$GPU_MEM_UTIL
SERVE_MODE=$SERVE_MODE
API_HOST=$API_HOST
API_PORT=$API_PORT
HF_CACHE_DIR=$HF_CACHE_DIR
NCCL_IB_DISABLE=$NCCL_IB_DISABLE
EOF
}

main() {
  parse_args "$@"
  if [ "$ACTION" = "up" ]; then
    check_host_preflight
    detect_local_fabric
    if [ "$ROLE_MODE" = "cluster" ]; then
      detect_remote_fabric
      verify_fabric_ping
    fi
  fi
  if [ "$DRY_RUN" = "1" ]; then
    print_config
    exit 0
  fi
}

main "$@"
