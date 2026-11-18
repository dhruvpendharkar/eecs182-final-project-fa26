#!/usr/bin/env bash
# Launch the EECS 182 dose-response study on a multi-GPU node (RTX 6000 class GPUs).
#
# Examples
#   scripts/launch.sh                                         # every stage, GPUs from configs/cluster.env
#   scripts/launch.sh --gpus 0,1,2,3                          # every stage, 4 GPUs
#   scripts/launch.sh --gpus 0,1 --stages prepare,train       # selected stages only
#   scripts/launch.sh --gpus 2,3 --model qwen1_5b \
#       --config configs/local-pilot.yaml --allow-unpinned    # pilot
#   scripts/launch.sh --judge-gpus 2 --detach                 # 2-GPU judge, survives logout
#   scripts/launch.sh --dry-run                               # print the job plan
#   scripts/launch.sh --status                                # progress and GPU hours
#
# Every option can also be set as an environment variable, e.g.
#   GPUS=0,1 STAGES=train,generate scripts/launch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Machine defaults (KEY=value lines). Environment variables and flags take precedence.
ENV_FILE="${ENV_FILE:-configs/cluster.env}"
if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r key value; do
    key="${key//[[:space:]]/}"
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    value="${value%%#*}"; value="${value%"${value##*[![:space:]]}"}"
    [[ -z "${!key+x}" ]] && export "$key=$value"
  done < "$ENV_FILE"
fi

GPUS="${GPUS:-}"                                   # empty: CUDA_VISIBLE_DEVICES, else every GPU
STAGES="${STAGES:-all}"                            # all or a comma list of the stages below
MODEL="${MODEL:-qwen7b}"
CONFIG="${CONFIG:-configs/study.yaml}"
BACKEND="${BACKEND:-vllm}"
VLLM_GPUS_PER_JOB="${VLLM_GPUS_PER_JOB:-1}"        # tensor parallel size for clean/generate
JUDGE_GPUS_PER_JOB="${JUDGE_GPUS_PER_JOB:-1}"      # tensor parallel size for the 32B judge
MAX_CPU_JOBS="${MAX_CPU_JOBS:-2}"
VENV="${VENV:-.venv}"
LOG_DIR="${LOG_DIR:-logs}"
DRY_RUN="${DRY_RUN:-0}"
DETACH="${DETACH:-0}"
ALLOW_UNPINNED="${ALLOW_UNPINNED:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
ARCHIVE_INTERRUPTED="${ARCHIVE_INTERRUPTED:-1}"
STATUS=0
VALID_STAGES="fetch prepare train generate score extract analysis"

usage() {
  cat <<EOF
Usage: scripts/launch.sh [options]

  -g, --gpus IDS          Comma-separated GPU ids, e.g. 0,1,2,3 (default: all visible)
  -s, --stages LIST       all, or a comma list of: ${VALID_STAGES// /, }
  -m, --model KEY         Model key in the config (default: qwen7b)
  -c, --config PATH       Study config (default: configs/study.yaml)
  -b, --backend NAME      vllm or transformers (default: vllm)
      --vllm-gpus N       GPUs per clean/generate job (default: 1)
      --judge-gpus N      GPUs per 32B judge job (default: 1; use 2 on 24 GB cards)
      --max-cpu-jobs N    Concurrent CPU-only jobs (default: 2)
      --venv PATH         Virtualenv to activate (default: .venv)
      --hf-home PATH      Hugging Face cache location (default: ~/.cache/huggingface)
      --allow-unpinned    Allow floating model revisions (pilots only)
      --fail-fast         Stop launching jobs after the first failure
      --no-archive        Refuse to retrain over a partial run instead of moving it aside
      --detach            Run under nohup and return; output goes to ${LOG_DIR}/
      --dry-run           Print the job plan without running anything
      --status            Show finished jobs and measured GPU hours, then exit
  -h, --help              Show this help
EOF
}

die() { echo "launch.sh: $*" >&2; exit 2; }
warn() { echo "launch.sh: warning: $*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpus) GPUS="${2:?$1 needs a value}"; shift 2 ;;
    -s|--stages) STAGES="${2:?$1 needs a value}"; shift 2 ;;
    -m|--model) MODEL="${2:?$1 needs a value}"; shift 2 ;;
    -c|--config) CONFIG="${2:?$1 needs a value}"; shift 2 ;;
    -b|--backend) BACKEND="${2:?$1 needs a value}"; shift 2 ;;
    --vllm-gpus) VLLM_GPUS_PER_JOB="${2:?$1 needs a value}"; shift 2 ;;
    --judge-gpus) JUDGE_GPUS_PER_JOB="${2:?$1 needs a value}"; shift 2 ;;
    --max-cpu-jobs) MAX_CPU_JOBS="${2:?$1 needs a value}"; shift 2 ;;
    --venv) VENV="${2:?$1 needs a value}"; shift 2 ;;
    --hf-home) export HF_HOME="${2:?$1 needs a value}"; shift 2 ;;
    --allow-unpinned) ALLOW_UNPINNED=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --no-archive) ARCHIVE_INTERRUPTED=0; shift ;;
    --detach) DETACH=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --status) STATUS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option $1" ;;
  esac
done

for value in "$VLLM_GPUS_PER_JOB" "$JUDGE_GPUS_PER_JOB" "$MAX_CPU_JOBS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "GPU and job counts must be positive integers, got '$value'"
done
[[ "$BACKEND" == vllm || "$BACKEND" == transformers ]] || die "--backend must be vllm or transformers"
[[ -f "$CONFIG" ]] || die "config $CONFIG not found"

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  warn "no virtualenv at $VENV; using the python on PATH"
fi
PYTHON="$(command -v python || command -v python3)" || die "python not found"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

if [[ "$STATUS" == 1 ]]; then
  exec "$PYTHON" run_study.py --config "$CONFIG" --model "$MODEL" status --backend "$BACKEND"
fi

STAGE_FLAGS=()
NEEDS_JUDGE=0
if [[ "$STAGES" == all ]]; then
  NEEDS_JUDGE=1
else
  IFS=',' read -r -a STAGE_LIST <<< "$STAGES"
  for stage in "${STAGE_LIST[@]}"; do
    [[ " $VALID_STAGES " == *" $stage "* ]] || die "unknown stage '$stage' (choose from: $VALID_STAGES)"
    STAGE_FLAGS+=(--stage "$stage")
    [[ "$stage" == score ]] && NEEDS_JUDGE=1
  done
fi

# ---- GPU checks ----------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  QUERY="index,name,memory.total,memory.used,compute_cap"
  TABLE="$(nvidia-smi --query-gpu="$QUERY" --format=csv,noheader,nounits 2>/dev/null \
           || nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits)"
  if [[ -z "$GPUS" ]]; then
    GPUS="${CUDA_VISIBLE_DEVICES:-$(cut -d, -f1 <<< "$TABLE" | tr -d ' ' | paste -sd, -)}"
  fi
  PRECISION="$("$PYTHON" -c 'import sys; from emdose.cli import load_config; print(load_config(sys.argv[1], sys.argv[2])["training"]["precision"])' "$CONFIG" "$MODEL")"
  MIN_MEM=999999
  echo "GPUs for this launch:"
  IFS=',' read -r -a GPU_LIST <<< "$GPUS"
  for gpu in "${GPU_LIST[@]}"; do
    row="$(awk -F', ' -v id="$gpu" '$1 == id' <<< "$TABLE")"
    if [[ -z "$row" ]]; then
      [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && continue   # ids may be remapped inside a scheduler allocation
      die "GPU $gpu not found by nvidia-smi"
    fi
    IFS=',' read -r index name total used cc <<< "$row"
    name="${name# }"; total="${total// /}"; used="${used// /}"; cc="${cc// /}"
    echo "  [$index] $name, ${total} MiB, ${used} MiB in use, compute ${cc:-?}"
    (( total < MIN_MEM )) && MIN_MEM=$total
    (( used > 2048 )) && warn "GPU $gpu already has ${used} MiB in use by another process"
    if [[ -n "$cc" && "${cc%%.*}" -lt 8 && "$PRECISION" == bf16 ]]; then
      die "GPU $gpu ($name) has no native bf16. Copy the config, set training.precision: fp16, and pass --config."
    fi
  done
  if [[ "$NEEDS_JUDGE" == 1 && "$JUDGE_GPUS_PER_JOB" == 1 && "$MIN_MEM" -lt 40000 ]]; then
    warn "the 32B AWQ judge needs about 20 GB for weights plus KV cache; on ${MIN_MEM} MiB cards use --judge-gpus 2"
  fi
elif [[ "$DRY_RUN" != 1 ]]; then
  die "nvidia-smi not found; run on a GPU node or use --dry-run"
fi

# Model weights (~40 GB) and run outputs need room; a full home quota can block logins.
if [[ "$DRY_RUN" != 1 && ( "$STAGES" == all || "$STAGES" == *fetch* || "$STAGES" == *prepare* || "$STAGES" == *train* ) ]]; then
  WORKDIR="$("$PYTHON" -c 'import sys; from emdose.cli import load_config; print(load_config(sys.argv[1], sys.argv[2])["workdir"])' "$CONFIG" "$MODEL")"
  "$PYTHON" - "${HF_HOME:-$HOME/.cache/huggingface}" "$WORKDIR" <<'EOF' || die "not enough free space (see above). Free some space and retry."
import os, shutil, sys
for label, path, need in (("Hugging Face cache", sys.argv[1], 60), ("workdir", sys.argv[2], 20)):
    existing = os.path.abspath(path)
    while not os.path.exists(existing):
        existing = os.path.dirname(existing)
    free = shutil.disk_usage(existing).free / 1e9
    print(f"  {'ok ' if free >= need else 'LOW'} {free:7.1f} GB free for the {label} ({path}), want {need} GB")
    if free < need:
        sys.exit(1)
EOF
fi

if [[ -z "${HF_TOKEN:-}" && ! -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
  warn "no Hugging Face token found. Private data sources and gated graders need 'hf auth login' or HF_TOKEN."
fi

# ---- Launch --------------------------------------------------------------------
CMD=("$PYTHON" run_study.py --config "$CONFIG" --model "$MODEL" run
     --backend "$BACKEND" --vllm-gpus-per-job "$VLLM_GPUS_PER_JOB"
     --judge-gpus-per-job "$JUDGE_GPUS_PER_JOB" --max-cpu-jobs "$MAX_CPU_JOBS")
[[ -n "$GPUS" ]] && CMD+=(--gpus "$GPUS")
CMD+=(${STAGE_FLAGS[@]+"${STAGE_FLAGS[@]}"})
[[ "$DRY_RUN" != 1 ]] && CMD+=(--execute)
[[ "$ALLOW_UNPINNED" == 1 ]] && CMD+=(--allow-unpinned)
[[ "$FAIL_FAST" == 1 ]] && CMD+=(--fail-fast)
[[ "$ARCHIVE_INTERRUPTED" == 1 ]] && CMD+=(--archive-interrupted)

if [[ "$DRY_RUN" == 1 ]]; then
  exec "${CMD[@]}"
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/launch-$MODEL-$(date +%Y%m%d-%H%M%S).log"
echo "Model $MODEL | config $CONFIG | stages $STAGES | GPUs ${GPUS:-none} | log $LOG"
if [[ "$DETACH" == 1 ]]; then
  nohup "${CMD[@]}" >"$LOG" 2>&1 </dev/null &
  echo "$!" > "$LOG_DIR/launch-$MODEL.pid"
  echo "Started PID $!. Follow with: tail -f $LOG   Stop with: kill \$(cat $LOG_DIR/launch-$MODEL.pid)"
else
  # tee -i keeps logging through Ctrl-C so the launcher can stop its jobs cleanly.
  "${CMD[@]}" 2>&1 | tee -i "$LOG"
fi
