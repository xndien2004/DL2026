#!/usr/bin/env bash
# Setup: install Python deps + download DetectDrill dataset from HuggingFace.
# Run once before baseline.sh or simam.sh.
#
# Usage:
#   bash download.sh               # install deps + download
#   bash download.sh skip-install  # only download (skip pip install)
#
# Env:
#   HF_TOKEN    HuggingFace token for private repos (optional)
#   HF_REPO     Override default repo  (default: nhonhoccode/DetectDrill)
#   PYTHON      Python executable      (default: python3)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

SKIP_INSTALL="${1:-}"
DATA_DIR="${SCRIPT_DIR}/data"
DATA_AUG_DIR="${DATA_DIR}/DataAug"
HF_REPO="${HF_REPO:-nhonhoccode/DetectDrill}"
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo " download.sh — setup + dataset download"
echo " Script dir : ${SCRIPT_DIR}"
echo " HF repo    : ${HF_REPO}"
echo "============================================================"

if [[ "${SKIP_INSTALL}" != "skip-install" ]]; then
    echo ">>> Installing Python dependencies"
    "${PYTHON}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

if [[ -d "${DATA_AUG_DIR}/train" ]]; then
    echo ">>> Dataset already present at ${DATA_AUG_DIR}"
else
    echo ">>> Downloading dataset from HuggingFace (${HF_REPO})"
    "${PYTHON}" "${SCRIPT_DIR}/download_data.py" \
        --repo   "${HF_REPO}" \
        --target "${DATA_DIR}"
fi
