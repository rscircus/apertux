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

HEAD_FABRIC_IFACE=""
HEAD_FABRIC_IP=""
RESOLVED_WORKER_IFACE=""
WORKER_FABRIC_IP=""
DOCKER_ENV_ARGS=()
MODEL_MOUNT_ARGS=()
CONTAINER_MODEL_REF=""
VLLM_SERVE_CMD=""

SPARK_CX7_IFACES="enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1"

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

  if [ "$ROLE_MODE" = "worker-only" ] && [ "$ACTION" = "up" ] && [ -z "$HEAD_IP" ]; then
    die "--worker-only requires --head-ip <spark1-200g-ipv4>"
  fi
}

check_host_preflight() {
  local arch
  arch="$(uname -m)"
  if [ "$arch" != "aarch64" ] && [ "$ALLOW_ARCH_MISMATCH" != "1" ]; then
    die "Expected aarch64 (DGX Spark GB10), got '$arch' (use --allow-arch-mismatch to override)"
  fi
  command -v docker >/dev/null 2>&1 || die "docker not found on local Spark"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found on local Spark"
  if [ "$DRY_RUN" != "1" ]; then
    docker info >/dev/null 2>&1 || die "docker daemon is not reachable on local Spark"
    nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed on local Spark"
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
  [ -n "$HEAD_FABRIC_IFACE" ] || die "No active ConnectX-7 200GbE port found on local Spark (check QSFP cable or pass --fabric-iface)"

  HEAD_FABRIC_IP="$(extract_ipv4 "$HEAD_FABRIC_IFACE")"
  if [ -z "$HEAD_FABRIC_IP" ] && [ "$ASSIGN_FABRIC_IPS" = "1" ] && [ "$DRY_RUN" != "1" ]; then
    local target_cidr="$HEAD_ASSIGN_CIDR"
    if [ "$ROLE_MODE" = "worker-only" ]; then
      target_cidr="$WORKER_ASSIGN_CIDR"
    fi
    sudo ip addr add "$target_cidr" dev "$HEAD_FABRIC_IFACE"
    sudo ip link set "$HEAD_FABRIC_IFACE" up
    HEAD_FABRIC_IP="$(extract_ipv4 "$HEAD_FABRIC_IFACE")"
  fi
  if [ -z "$HEAD_FABRIC_IP" ]; then
    die "Interface $HEAD_FABRIC_IFACE has no IPv4. Run 'sudo ip addr add $HEAD_ASSIGN_CIDR dev $HEAD_FABRIC_IFACE' or pass --assign-fabric-ips"
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

model_cache_subdir() {
  local m="$1"
  printf 'models--%s\n' "${m//\//--}"
}

has_local_weights() {
  local model_id="$1"
  local cache_dir="$2"
  if [ -d "$model_id" ] && compgen -G "$model_id/*.safetensors" >/dev/null; then
    return 0
  fi
  local sub
  sub="$(model_cache_subdir "$model_id")"
  if compgen -G "$cache_dir/hub/$sub/snapshots/*/*.safetensors" >/dev/null; then
    return 0
  fi
  return 1
}

has_local_image() {
  docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1
}

confirm_bootstrap_or_die() {
  local reason="$1"
  if [ "$APPROVE_DOWNLOAD" = "1" ]; then
    return 0
  fi
  if [ "${APERTUX_DISABLE_TTY_PROMPT:-0}" != "1" ] && [ -r /dev/tty ] && [ -w /dev/tty ]; then
    printf 'Bootstrap required (%s).\nAuthorize first-time download to local cache? [y/N]: ' "$reason" >/dev/tty
    local reply=""
    read -r reply </dev/tty || true
    case "$reply" in
      y|Y|yes|YES)
        APPROVE_DOWNLOAD=1
        return 0
        ;;
    esac
  fi
  die "Missing local artifacts ($reason). Local-only rule blocks unapproved downloads; pass --approve-download to authorize first-time cache population."
}

download_local_weights() {
  mkdir -p "$HF_CACHE_DIR"
  local token_args=()
  if [ -n "${HF_TOKEN:-}" ]; then
    token_args=(-e HF_TOKEN)
  fi
  docker run --rm \
    -e HF_HUB_DISABLE_TELEMETRY=1 \
    -e DO_NOT_TRACK=1 \
    "${token_args[@]}" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    "$VLLM_IMAGE" \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"
}

ensure_artifacts_ready() {
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  local head_img_ok=0 head_wt_ok=0 worker_img_ok=1 worker_wt_ok=1
  has_local_image && head_img_ok=1 || true
  has_local_weights "$MODEL" "$HF_CACHE_DIR" && head_wt_ok=1 || true

  if [ "$ROLE_MODE" = "cluster" ]; then
    local out
    out="$(ssh "$WORKER_SSH" bash -s -- "$VLLM_IMAGE" "$MODEL" <<'REMOTE_EOF'
set -euo pipefail
# ROLE=check_cache
img="$1"
mdl="$2"
cdir="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
img_ok=0
wt_ok=0
docker image inspect "$img" >/dev/null 2>&1 && img_ok=1 || true
sub="models--${mdl//\//--}"
if [ -d "$mdl" ] && compgen -G "$mdl/*.safetensors" >/dev/null; then
  wt_ok=1
elif compgen -G "$cdir/hub/$sub/snapshots/*/*.safetensors" >/dev/null; then
  wt_ok=1
fi
printf 'IMAGE_OK=%s\nWEIGHTS_OK=%s\n' "$img_ok" "$wt_ok"
REMOTE_EOF
)"
    worker_img_ok="$(printf '%s\n' "$out" | awk -F= '/^IMAGE_OK=/ { print $2; exit }')"
    worker_wt_ok="$(printf '%s\n' "$out" | awk -F= '/^WEIGHTS_OK=/ { print $2; exit }')"
  fi

  local missing=()
  [ "$head_img_ok" = "1" ] || missing+=("Spark1:image:$VLLM_IMAGE")
  [ "$head_wt_ok" = "1" ] || missing+=("Spark1:weights:$MODEL")
  [ "$worker_img_ok" = "1" ] || missing+=("Spark2:image:$VLLM_IMAGE")
  [ "$worker_wt_ok" = "1" ] || missing+=("Spark2:weights:$MODEL")

  if [ "${#missing[@]}" -eq 0 ]; then
    return 0
  fi

  confirm_bootstrap_or_die "${missing[*]}"

  if [ "$head_img_ok" != "1" ]; then
    docker pull "$VLLM_IMAGE"
  fi
  if [ "$head_wt_ok" != "1" ]; then
    download_local_weights
  fi

  if [ "$ROLE_MODE" = "cluster" ] && { [ "$worker_img_ok" != "1" ] || [ "$worker_wt_ok" != "1" ]; }; then
    ssh "$WORKER_SSH" bash -s -- "$VLLM_IMAGE" "$MODEL" "$worker_img_ok" "$worker_wt_ok" <<REMOTE_EOF
set -euo pipefail
# ROLE=bootstrap
img="\$1"
mdl="\$2"
img_ok="\$3"
wt_ok="\$4"
cdir="\${HF_CACHE_DIR:-\$HOME/.cache/huggingface}"
mkdir -p "\$cdir"
if [ "\$img_ok" != "1" ]; then
  docker pull "\$img"
fi
if [ "\$wt_ok" != "1" ]; then
  export HF_TOKEN="${HF_TOKEN:-}"
  token_flag=()
  if [ -n "\$HF_TOKEN" ]; then
    token_flag=(-e HF_TOKEN)
  fi
  docker run --rm \
    -e HF_HUB_DISABLE_TELEMETRY=1 \
    -e DO_NOT_TRACK=1 \
    "\${token_flag[@]}" \
    -v "\$cdir:/root/.cache/huggingface" \
    "\$img" \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('\$mdl')"
fi
REMOTE_EOF
  fi
}

build_common_docker_env() {
  local node_ip="$1"
  local node_if="$2"
  DOCKER_ENV_ARGS=(
    -e "VLLM_HOST_IP=$node_ip"
    -e "NCCL_SOCKET_IFNAME=$node_if"
    -e "GLOO_SOCKET_IFNAME=$node_if"
    -e "TP_SOCKET_IFNAME=$node_if"
    -e "UCX_NET_DEVICES=$node_if"
    -e "HF_HUB_OFFLINE=1"
    -e "TRANSFORMERS_OFFLINE=1"
    -e "VLLM_NO_USAGE_STATS=1"
    -e "DO_NOT_TRACK=1"
    -e "HF_HUB_DISABLE_TELEMETRY=1"
    -e "RAY_USAGE_STATS_ENABLED=0"
  )
  if [ "$NCCL_IB_DISABLE" = "1" ]; then
    DOCKER_ENV_ARGS+=(-e "NCCL_IB_DISABLE=1")
  fi
}

build_model_mount_args() {
  MODEL_MOUNT_ARGS=(-v "$HF_CACHE_DIR:/root/.cache/huggingface")
  CONTAINER_MODEL_REF="$MODEL"
  if [ -d "$MODEL" ]; then
    local abs_model
    abs_model="$(cd "$MODEL" && pwd)"
    MODEL_MOUNT_ARGS+=(-v "$abs_model:/models/target:ro")
    CONTAINER_MODEL_REF="/models/target"
  fi
}

start_ray_head() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  build_common_docker_env "$HEAD_FABRIC_IP" "$HEAD_FABRIC_IFACE"
  build_model_mount_args
  mkdir -p "$HF_CACHE_DIR"
  docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc host \
    --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${DOCKER_ENV_ARGS[@]}" \
    "${MODEL_MOUNT_ARGS[@]}" \
    --entrypoint bash \
    "$VLLM_IMAGE" \
    -lc "ray start --head --node-ip-address=$HEAD_FABRIC_IP --port=$RAY_PORT --disable-usage-stats --block" >/dev/null
}

start_local_worker() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  build_common_docker_env "$HEAD_FABRIC_IP" "$HEAD_FABRIC_IFACE"
  build_model_mount_args
  mkdir -p "$HF_CACHE_DIR"
  docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc host \
    --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${DOCKER_ENV_ARGS[@]}" \
    "${MODEL_MOUNT_ARGS[@]}" \
    --entrypoint bash \
    "$VLLM_IMAGE" \
    -lc "ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$HEAD_FABRIC_IP --disable-usage-stats --block" >/dev/null
}

start_remote_worker() {
  ssh "$WORKER_SSH" bash -s -- \
    "$CONTAINER_NAME" \
    "$VLLM_IMAGE" \
    "$HEAD_FABRIC_IP" \
    "$RAY_PORT" \
    "$WORKER_FABRIC_IP" \
    "$RESOLVED_WORKER_IFACE" \
    "$NCCL_IB_DISABLE" \
    "$MODEL" <<'REMOTE_EOF'
set -euo pipefail
# ROLE=start_worker
cname="$1"
img="$2"
head_ip="$3"
rport="$4"
wip="$5"
wif="$6"
ib_dis="$7"
mdl="$8"
cdir="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
mkdir -p "$cdir"
docker rm -f "$cname" >/dev/null 2>&1 || true
extra_env=()
if [ "$ib_dis" = "1" ]; then
  extra_env+=(-e "NCCL_IB_DISABLE=1")
fi
extra_mounts=()
if [ -d "$mdl" ]; then
  abs_mdl="$(cd "$mdl" && pwd)"
  extra_mounts+=(-v "$abs_mdl:/models/target:ro")
fi
docker run -d \
  --name "$cname" \
  --network host \
  --ipc host \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e "VLLM_HOST_IP=$wip" \
  -e "NCCL_SOCKET_IFNAME=$wif" \
  -e "GLOO_SOCKET_IFNAME=$wif" \
  -e "TP_SOCKET_IFNAME=$wif" \
  -e "UCX_NET_DEVICES=$wif" \
  -e "HF_HUB_OFFLINE=1" \
  -e "TRANSFORMERS_OFFLINE=1" \
  -e "VLLM_NO_USAGE_STATS=1" \
  -e "DO_NOT_TRACK=1" \
  -e "HF_HUB_DISABLE_TELEMETRY=1" \
  -e "RAY_USAGE_STATS_ENABLED=0" \
  "${extra_env[@]}" \
  -v "$cdir:/root/.cache/huggingface" \
  "${extra_mounts[@]}" \
  --entrypoint bash \
  "$img" \
  -lc "ray start --address=$head_ip:$rport --node-ip-address=$wip --disable-usage-stats --block" >/dev/null
REMOTE_EOF
}

wait_for_ray_cluster() {
  local max_tries="${APERTUX_RAY_WAIT_TRIES:-30}"
  local i=1 status_out=""
  while [ "$i" -le "$max_tries" ]; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || printf 'false')" != "true" ]; then
      docker logs --tail 40 "$CONTAINER_NAME" >&2 || true
      die "Head container $CONTAINER_NAME exited unexpectedly while starting Ray"
    fi
    status_out="$(docker exec "$CONTAINER_NAME" ray status 2>/dev/null || true)"
    if printf '%s\n' "$status_out" | grep -Eq '2\.0.*GPU|[2-9]\.0/[2-9]\.0 GPU'; then
      return 0
    fi
    sleep "${APERTUX_POLL_INTERVAL:-1}"
    i=$((i + 1))
  done
  printf '%s\n' "$status_out" >&2
  die "Timed out waiting for 2 GPUs to register in Ray cluster across Spark 1 and Spark 2"
}

build_vllm_serve_cmd() {
  build_model_mount_args
  local args=(
    vllm serve "$CONTAINER_MODEL_REF"
    --served-model-name "$MODEL"
    --host "$API_HOST"
    --port "$API_PORT"
    --tensor-parallel-size 2
    --pipeline-parallel-size 1
    --distributed-executor-backend ray
    --dtype bfloat16
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --chat-template-content-format string
    --compilation-config.pass_config.fuse_allreduce_rms false
  )
  if [ "$KV_CACHE_DTYPE" != "auto" ]; then
    args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  fi
  if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    args+=(--enable-prefix-caching)
  fi
  case "$SERVE_MODE" in
    tools)
      args+=(--enable-auto-tool-choice --tool-call-parser apertus)
      ;;
    thinking)
      args+=(--reasoning-parser apertus --default-chat-template-kwargs.enable_thinking true)
      ;;
    plain)
      ;;
  esac
  VLLM_SERVE_CMD="${args[*]}"
}

start_vllm_serve() {
  build_vllm_serve_cmd
  docker exec -d "$CONTAINER_NAME" bash -lc "$VLLM_SERVE_CMD >/tmp/vllm-serve.log 2>&1"
}

fetch_vllm_log_tail() {
  local lines="${1:-40}"
  docker exec "$CONTAINER_NAME" bash -lc "tail -n $lines /tmp/vllm-serve.log 2>/dev/null || true" 2>/dev/null || true
}

wait_for_vllm_ready() {
  local max_tries="${APERTUX_HEALTH_WAIT_TRIES:-180}"
  local interval="${APERTUX_POLL_INTERVAL:-2}"
  local i=1 log_tail=""
  while [ "$i" -le "$max_tries" ]; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || printf 'false')" != "true" ]; then
      docker logs --tail 40 "$CONTAINER_NAME" >&2 || true
      die "Container $CONTAINER_NAME stopped while loading $MODEL"
    fi
    log_tail="$(fetch_vllm_log_tail 60)"
    if printf '%s\n' "$log_tail" | grep -Eq 'Traceback \(most recent call last\)|OutOfMemoryError|RuntimeError:|NCCL ERROR'; then
      printf '%s\n' "$log_tail" >&2
      die "Fatal error detected in /tmp/vllm-serve.log during $MODEL startup"
    fi
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if [ "$interval" != "0" ]; then
      sleep "$interval"
    fi
    i=$((i + 1))
  done
  fetch_vllm_log_tail 60 >&2
  die "Timed out waiting for vLLM /health on port $API_PORT"
}

cmd_smoke() {
  local endpoint="${SMOKE_ENDPOINT:-http://127.0.0.1:${API_PORT}}"
  endpoint="${endpoint%/}"
  local payload resp
  payload="$(printf '{"model":"%s","messages":[{"role":"user","content":"Reply in one short sentence confirming Apertus is online."}],"max_tokens":64,"temperature":0}' "$MODEL")"
  resp="$(curl -fsS "${endpoint}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload")" || die "Smoke check failed against ${endpoint}/v1/chat/completions"
  printf '%s\n' "$resp" | grep -Fq '"choices"' || die "Unexpected response from ${endpoint}/v1/chat/completions: $resp"
  printf 'Smoke check OK: %s\n' "$resp"
}

cmd_status() {
  local running
  running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || printf 'false')"
  if [ "$running" != "true" ]; then
    die "Head container $CONTAINER_NAME is not running"
  fi
  if [ -n "$WORKER_SSH" ]; then
    local wrun
    wrun="$(ssh "$WORKER_SSH" bash -s -- "$CONTAINER_NAME" <<'REMOTE_EOF'
set -euo pipefail
# ROLE=status_worker
docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || printf 'false'
REMOTE_EOF
)"
    [ "$wrun" = "true" ] || die "Worker container $CONTAINER_NAME on $WORKER_SSH is not running"
  fi
  printf '=== Ray Cluster ===\n'
  docker exec "$CONTAINER_NAME" ray status
  printf '=== HTTP Health ===\n'
  curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null && printf 'HTTP :%s/health OK\n' "$API_PORT"
  printf '=== Recent vLLM Logs ===\n'
  fetch_vllm_log_tail 20
}

cmd_logs() {
  if [ "$FOLLOW_LOGS" = "1" ]; then
    docker exec "$CONTAINER_NAME" bash -lc "tail -n 100 -f /tmp/vllm-serve.log"
  else
    fetch_vllm_log_tail 100
  fi
}

cmd_down() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if [ -n "$WORKER_SSH" ]; then
    ssh "$WORKER_SSH" bash -s -- "$CONTAINER_NAME" <<'REMOTE_EOF'
set -euo pipefail
# ROLE=stop_worker
docker rm -f "$1" >/dev/null 2>&1 || true
REMOTE_EOF
  fi
  printf 'Stopped %s.\n' "$CONTAINER_NAME"
}

print_config() {
  build_vllm_serve_cmd
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
VLLM_SERVE_CMD=$VLLM_SERVE_CMD
EOF
}

print_client_curl_hint() {
  local host_hint="${HEAD_FABRIC_IP:-127.0.0.1}"
  cat <<EOF

Apertus is live across 2x DGX Spark (TP=2). Query from any reachable host:
curl -sS http://${host_hint}:${API_PORT}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"${MODEL}","messages":[{"role":"user","content":"Gruezi! Explain Switzerland in two sentences."}],"max_tokens":128}'
EOF
}

cmd_up() {
  check_host_preflight
  detect_local_fabric
  if [ "$ROLE_MODE" = "cluster" ]; then
    detect_remote_fabric
    verify_fabric_ping
  fi
  if [ "$DRY_RUN" = "1" ]; then
    print_config
    return 0
  fi
  ensure_artifacts_ready

  if [ "$ROLE_MODE" = "worker-only" ]; then
    start_local_worker
    printf 'Ray worker started on %s (%s) -> %s:%s\n' "$HEAD_FABRIC_IFACE" "$HEAD_FABRIC_IP" "$HEAD_IP" "$RAY_PORT"
    return 0
  fi

  start_ray_head
  if [ "$ROLE_MODE" = "cluster" ]; then
    start_remote_worker
  fi
  wait_for_ray_cluster
  start_vllm_serve
  wait_for_vllm_ready
  cmd_smoke
  print_client_curl_hint
}

main() {
  parse_args "$@"
  if [ "$DRY_RUN" = "1" ] && [ "$ACTION" != "up" ]; then
    print_config
    exit 0
  fi
  case "$ACTION" in
    up)
      cmd_up
      ;;
    down)
      cmd_down
      ;;
    status)
      cmd_status
      ;;
    smoke)
      cmd_smoke
      ;;
    logs)
      cmd_logs
      ;;
  esac
}

main "$@"
