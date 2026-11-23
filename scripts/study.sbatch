#!/bin/bash
# Run the study as a SLURM batch job, which keeps running after you log out
# (processes started in an SSH session on this cluster end with the session).
#
#   sbatch scripts/study.sbatch                         # full 7B study on 6 GPUs
#   sbatch -p PARTITION -w d6 scripts/study.sbatch      # pick a partition / node if your cluster needs it
#   CONFIG=configs/size-1_5b.yaml MODEL=qwen1_5b sbatch --export=ALL scripts/study.sbatch
#
# Submit from the repository root. Progress: tail -f logs/slurm-<jobid>.log, or
# scripts/launch.sh --config configs/study.yaml --status. Cancel: scancel <jobid>.
# If the job hits its time limit, SLURM sends SIGTERM; the launcher stops its jobs
# cleanly, and resubmitting resumes where it stopped.
#SBATCH --job-name=emdose-study
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.log
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck disable=SC1091
source .venv/bin/activate
CONFIG="${CONFIG:-configs/study.yaml}"
MODEL="${MODEL:-qwen7b}"
STAGES="${STAGES:-fetch,prepare,train,generate,score,extract,analysis}"
GPUS="${CUDA_VISIBLE_DEVICES:?SLURM did not assign GPUs; request them with --gres=gpu:N}"
echo "$(date '+%F %T') job ${SLURM_JOB_ID:-?} on $(hostname): GPUs ${GPUS}, config ${CONFIG}, stages ${STAGES}"
# The launcher gives each job its own GPUs from the ones SLURM assigned.
exec scripts/launch.sh --config "$CONFIG" --model "$MODEL" --gpus "$GPUS" --stages "$STAGES"
