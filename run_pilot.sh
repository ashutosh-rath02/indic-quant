#!/usr/bin/env bash
# End-to-end pilot run. Resumable: every step skips outputs that already exist.
#
#   ./run_pilot.sh                     # full run with config.yaml
#   ./run_pilot.sh --dry-run           # fake tools and data, finishes in under a minute
#   CONFIG=configs/qwen3-4b.yaml ./run_pilot.sh
#   ./run_pilot.sh --from run_kld      # restart from a given step
#
# Logs: logs/<step>.log plus per-command logs under logs/<step>/.
set -euo pipefail

cd "$(dirname "$0")"
CONFIG="${CONFIG:-config.yaml}"
VENV="${VENV:-.venv}"
PY="${PYTHON:-python3}"

if [ ! -x "$VENV/bin/python" ]; then
    "$PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -r requirements.txt
fi

if [[ " $* " != *" --dry-run "* ]]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "HF_TOKEN is not set. Gemma 3, FLORES+, IN22-Gen and MILU are gated on Hugging Face." >&2
        echo "Accept their terms on huggingface.co, then: export HF_TOKEN=hf_..." >&2
        exit 1
    fi
fi

exec "$VENV/bin/python" -m iqp all --config "$CONFIG" "$@"
