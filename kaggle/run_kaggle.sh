#!/usr/bin/env bash
# Kaggle session runner, equivalent to kaggle/indic_quant_pilot.ipynb.
#
# Run from a Kaggle notebook cell after cloning the repository:
#   !bash /kaggle/working/indic-quant/kaggle/run_kaggle.sh
#
# Environment (all optional):
#   IQP_CONFIG        config file, default configs/kaggle.yaml (or configs/kaggle-qwen3-4b.yaml)
#   IQP_MAX_HOURS     time budget for the whole session, default 10.5 (Kaggle stops at 12)
#   IQP_WORK_DIR      scratch directory for large files, default /tmp/iqp-work
#   IQP_OUTPUT_ROOT   persisted output, default /kaggle/working/iqp
#   IQP_RESTORE_FROM  previous session's output, default /kaggle/input (searched recursively)
#   IQP_CUDA_ARCHS    CUDA architectures to build, default "60;75" (P100 and T4)
#   IQP_RUN_TESTS     1 to run the unit tests first (about 2 minutes on 4 vCPUs)
#   HF_TOKEN          Hugging Face token; read from Kaggle Secrets when not set
#
# Every step is resumable. A session that ends early, or is stopped by the time budget, leaves
# a restorable output in $IQP_OUTPUT_ROOT; attach it to the next session and run this again.
set -euo pipefail

export IQP_SESSION_START="${IQP_SESSION_START:-$(date +%s)}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
CONFIG="${IQP_CONFIG:-configs/kaggle.yaml}"
MAX_HOURS="${IQP_MAX_HOURS:-10.5}"
PY="${PYTHON:-python3}"

echo "== environment"
nvidia-smi || echo "WARNING: nvidia-smi failed; select a GPU accelerator in the notebook settings"
df -h /kaggle/working "${IQP_WORK_DIR:-/tmp}" 2>/dev/null || df -h
if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -n 2
elif [ -x /usr/local/cuda/bin/nvcc ]; then /usr/local/cuda/bin/nvcc --version | tail -n 2
else echo "ERROR: nvcc not found. The CUDA build of llama.cpp needs it; use a GPU accelerator." >&2; exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN="$("$PY" -c 'from kaggle_secrets import UserSecretsClient; print(UserSecretsClient().get_secret("HF_TOKEN"))' 2>/dev/null || true)"
    export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set and could not be read from Kaggle Secrets (Add-ons > Secrets)." >&2
    exit 1
fi

echo "== python requirements"
"$PY" -m pip install -q -r requirements.txt

if [ "${IQP_RUN_TESTS:-0}" = "1" ]; then
    "$PY" -m pytest -q
fi

run() { echo "== $*"; "$PY" -m iqp "$@" --config "$CONFIG"; }

run restore
run status
run setup
run download
run convert
run validate_data
run prepare_data
run build_imatrix
run stream --max-hours "$MAX_HOURS"
run aggregate
run plot
run report
run bundle
run status
echo "== done. Results: ${IQP_OUTPUT_ROOT:-/kaggle/working/iqp}/results/REPORT.md"
