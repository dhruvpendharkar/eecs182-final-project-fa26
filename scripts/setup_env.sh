#!/usr/bin/env bash
# Create the project virtualenv on the GPU host. Run once from any directory:
#
#   scripts/setup_env.sh                  # .venv with every extra (GPU host)
#   PYTHON=python3.11 scripts/setup_env.sh
#   scripts/setup_env.sh --cpu            # data, demo and analysis only (no torch/vLLM)
#
# On clusters with environment modules, load a Python 3.10-3.12 module first
# (for example `module load python/3.11`). vLLM installs its own matching torch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-.venv}"
EXTRAS="train,inference,analysis,benchmarks,fetch"
[[ "${1:-}" == "--cpu" ]] && EXTRAS="analysis,fetch"

# Pick the first Python 3.10-3.12 available unless PYTHON is set.
if [[ -z "${PYTHON:-}" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))'; then
      PYTHON="$candidate"; break
    fi
  done
fi
[[ -n "${PYTHON:-}" ]] || { echo "setup_env.sh: need Python 3.10-3.12 (set PYTHON=...)" >&2; exit 2; }
echo "Using $("$PYTHON" --version) at $(command -v "$PYTHON")"

[[ -d "$VENV" ]] || "$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[${EXTRAS}]"
mkdir -p data logs
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

Environment ready in $VENV (recorded in data/environment.txt).
Next:
  source $VENV/bin/activate          # puts the emdose command on PATH
  hf auth login                      # or export HF_TOKEN=... (needed for the gated StrongREJECT grader)
  scripts/launch.sh --config configs/endpoints-7b.yaml --stages fetch
EOF
