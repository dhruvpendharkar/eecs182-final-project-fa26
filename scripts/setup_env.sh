#!/usr/bin/env bash
# Create the project virtualenv (.venv in the repo) on the GPU host.
#
#   scripts/setup_env.sh          # every extra (GPU host)
#   scripts/setup_env.sh --cpu    # data, demo and analysis only (no torch/vLLM)
#
# Safeguards against hangs: free space is checked before installing, pip runs at
# low CPU and I/O priority, stalled downloads time out and retry, and all output
# goes to logs/setup-<time>.log. On SLURM clusters, run this inside an
# allocation (srun --pty -c 8 --mem 32G scripts/setup_env.sh), not on a login
# node. Load a Python 3.10-3.12 module first if your cluster uses modules.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-.venv}"
CPU_ONLY=0
die() { echo "setup_env.sh: $*" >&2; exit 2; }
warn() { echo "setup_env.sh: warning: $*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) CPU_ONLY=1; shift ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

if [[ "$CPU_ONLY" == 1 ]]; then
  EXTRAS="analysis,fetch"; NEED_VENV="${MIN_FREE_GB:-3}"; NEED_CACHE=1; NEED_TMP=1
else
  # torch, vLLM and the CUDA libraries take about 12 GB installed, with
  # roughly 6 GB each of downloaded wheels and unpacked temporary files.
  EXTRAS="train,inference,analysis,benchmarks,fetch"; NEED_VENV="${MIN_FREE_GB:-15}"; NEED_CACHE=8; NEED_TMP=8
fi
PIP_CACHE="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
TMP="${TMPDIR:-/tmp}"
mkdir -p logs data

LOG="logs/setup-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"

# Pick the first Python 3.10-3.12 available unless PYTHON is set.
if [[ -z "${PYTHON:-}" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))'; then
      PYTHON="$candidate"; break
    fi
  done
fi
[[ -n "${PYTHON:-}" ]] || die "need Python 3.10-3.12 (set PYTHON=... or load a module)"
echo "Using $("$PYTHON" --version) at $(command -v "$PYTHON")"

if command -v sinfo >/dev/null 2>&1 && [[ -z "${SLURM_JOB_ID:-}" ]]; then
  warn "SLURM is available but this shell is not in an allocation. Installing on a login node can make it"
  warn "unresponsive for everyone. Prefer: srun --pty -c 8 --mem 32G scripts/setup_env.sh $*"
fi

echo "Checking free space:"
"$PYTHON" - "$VENV" "$PIP_CACHE" "$TMP" "$NEED_VENV" "$NEED_CACHE" "$NEED_TMP" <<'EOF' ||
import os, shutil, sys
paths, needs = sys.argv[1:4], [float(value) for value in sys.argv[4:7]]
groups = {}
for label, path, need in zip(("virtualenv", "pip cache", "temporary files"), paths, needs):
    existing = os.path.abspath(path)
    while not os.path.exists(existing):
        existing = os.path.dirname(existing)
    group = groups.setdefault(os.stat(existing).st_dev, {"path": existing, "need": 0.0, "uses": []})
    group["need"] += need
    group["uses"].append(f"{label} {path}")
short = False
for group in groups.values():
    free = shutil.disk_usage(group["path"]).free / 1e9
    short |= free < group["need"]
    print(f"  {'ok ' if free >= group['need'] else 'LOW'} {free:7.1f} GB free, need {group['need']:.0f} GB: "
          + "; ".join(group["uses"]))
sys.exit(1 if short else 0)
EOF
  die "not enough free space. Free some space (for example rm -rf ~/.cache/pip) and retry."
if command -v quota >/dev/null 2>&1; then
  echo "Your quotas (a full home quota can block logins):"
  quota -s 2>/dev/null | sed 's/^/  /' || true
fi

# Low priority keeps the node responsive; timeouts turn stalled downloads into errors.
LOW_PRIORITY=(nice -n 10)
command -v ionice >/dev/null 2>&1 && LOW_PRIORITY+=(ionice -c2 -n7)
export PIP_NO_INPUT=1 PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=5 PIP_PROGRESS_BAR=off MAX_JOBS="${MAX_JOBS:-4}"

[[ -d "$VENV" ]] || "$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
"${LOW_PRIORITY[@]}" python -m pip install --upgrade pip
"${LOW_PRIORITY[@]}" python -m pip install -e ".[${EXTRAS}]"
python -m pip freeze > data/environment.txt

python - <<'EOF'
import importlib.util
for name in ("torch", "vllm", "transformers", "peft", "sklearn", "strong_reject", "huggingface_hub"):
    print(f"  {name:<16} {'ok' if importlib.util.find_spec(name) else 'not installed'}")
try:
    import torch
    print(f"  CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
except ImportError:
    pass
EOF

cat <<EOF

Environment ready in $VENV (packages recorded in data/environment.txt, log in $LOG).
Next:
  source $VENV/bin/activate          # puts the emdose command on PATH
  hf auth login                      # or export HF_TOKEN=... (needed for the gated StrongREJECT grader)
  scripts/launch.sh --config configs/endpoints-7b.yaml --stages fetch
EOF
