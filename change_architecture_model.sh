#!/usr/bin/env bash
# End-to-end driver for the YOLOv12m + SimAM (V4) pipeline.
#
# Layout (all relative to this script's directory):
#   ./data/DataAug                          <- dataset (auto-downloaded from HF)
#   ./weight/best.pt                        <- pretrained checkpoint to fine-tune from
#   ./yolo_dataset_<variant>                <- prepared YOLO-flat layout (per variant)
#   ./runs/yolov12m-simam_<variant>/...     <- ultralytics training output
#
# Usage:
#   bash change_architecture_model.sh                          # full + mix
#   bash change_architecture_model.sh full mix
#   bash change_architecture_model.sh full bright
#   bash change_architecture_model.sh full dark
#   bash change_architecture_model.sh full all                 # iterate mix, bright, dark
#   bash change_architecture_model.sh skip-install bright      # skip pip install
#   bash change_architecture_model.sh eval-only dark           # eval only
#   bash change_architecture_model.sh pseudo mix               # full + pseudo-label retrain
#
# Args:
#   $1 MODE     full | skip-install | eval-only | pseudo   (default: full)
#   $2 VARIANT  mix  | bright | dark | all                  (default: mix)
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

MODE="${1:-full}"
VARIANT_ARG="${2:-mix}"

DATA_DIR="${SCRIPT_DIR}/data"
DATA_AUG_DIR="${DATA_DIR}/DataAug"
WEIGHT_DIR="${SCRIPT_DIR}/weight"
WEIGHT_FILE="${WEIGHT_DIR}/best.pt"
RUNS_PARENT="${SCRIPT_DIR}/runs"
HF_REPO="${HF_REPO:-nhonhoccode/DetectDrill}"

PYTHON="${PYTHON:-python}"

case "${VARIANT_ARG}" in
    mix|bright|dark)  VARIANTS=("${VARIANT_ARG}") ;;
    all)              VARIANTS=("mix" "bright" "dark") ;;
    *) echo "ERROR: unknown variant '${VARIANT_ARG}'. Use mix|bright|dark|all." >&2; exit 2 ;;
esac

case "${MODE}" in
    full|skip-install|eval-only|pseudo) ;;
    *) echo "ERROR: unknown mode '${MODE}'. Use full|skip-install|eval-only|pseudo." >&2; exit 2 ;;
esac

echo "============================================================"
echo " [V4 / SimAM] Working dir: ${SCRIPT_DIR}"
echo " Mode:        ${MODE}"
echo " Variant(s):  ${VARIANTS[*]}"
echo " Data:        ${DATA_AUG_DIR}"
echo " Weights:     ${WEIGHT_FILE}"
echo " HF repo:     ${HF_REPO}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
if [[ "${MODE}" != "skip-install" && "${MODE}" != "eval-only" ]]; then
    echo ">>> Installing Python dependencies"
    "${PYTHON}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

# ---------------------------------------------------------------------------
# 2. Download dataset if missing
# ---------------------------------------------------------------------------
if [[ ! -d "${DATA_AUG_DIR}" ]]; then
    echo ">>> Downloading dataset from HuggingFace (${HF_REPO})"
    "${PYTHON}" "${SCRIPT_DIR}/download_data.py" --repo "${HF_REPO}" --target "${DATA_DIR}"
else
    echo ">>> Dataset already present at ${DATA_AUG_DIR}"
fi

mkdir -p "${WEIGHT_DIR}"

# ---------------------------------------------------------------------------
# Per-variant pipeline
# ---------------------------------------------------------------------------
run_variant() {
    local VARIANT="$1"
    local WORK_DIR="${SCRIPT_DIR}/yolo_dataset_${VARIANT}"
    local DEFAULT_RUN_NAME="yolov12m-simam"
    local DEFAULT_RUN_DIR="${RUNS_PARENT}/${DEFAULT_RUN_NAME}"
    local RUN_NAME="${DEFAULT_RUN_NAME}_${VARIANT}"
    local RUN_DIR="${RUNS_PARENT}/${RUN_NAME}"
    local DEFAULT_PSEUDO_RUN_DIR="${RUNS_PARENT}/${DEFAULT_RUN_NAME}_pseudo"
    local PSEUDO_RUN_DIR="${RUNS_PARENT}/${RUN_NAME}_pseudo"

    echo
    echo "------------------------------------------------------------"
    echo " VARIANT = ${VARIANT}"
    echo " work_dir = ${WORK_DIR}"
    echo " run_dir  = ${RUN_DIR}"
    echo "------------------------------------------------------------"

    local PRETRAINED_FLAG=()
    if [[ -f "${WEIGHT_FILE}" ]]; then
        PRETRAINED_FLAG=(--pretrained "${WEIGHT_FILE}")
    fi

    # ---- Train ------------------------------------------------------------
    if [[ "${MODE}" != "eval-only" ]]; then
        # Wipe the default run dir so this variant starts clean.
        if [[ -d "${DEFAULT_RUN_DIR}" && "${DEFAULT_RUN_DIR}" != "${RUN_DIR}" ]]; then
            rm -rf "${DEFAULT_RUN_DIR}"
        fi

        echo ">>> [${VARIANT}] Training (run_train.py)"
        "${PYTHON}" "${SCRIPT_DIR}/run_train.py" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant "${VARIANT}" \
            "${PRETRAINED_FLAG[@]}"

        # Rename runs/yolov12m-simam -> runs/yolov12m-simam_<variant>
        if [[ -d "${DEFAULT_RUN_DIR}" ]]; then
            rm -rf "${RUN_DIR}"
            mv "${DEFAULT_RUN_DIR}" "${RUN_DIR}"
            echo ">>> [${VARIANT}] Saved run -> ${RUN_DIR}"
        fi
    fi

    # ---- Evaluate ---------------------------------------------------------
    echo ">>> [${VARIANT}] Evaluating (run_evaluate.py)"
    local EVAL_CKPT=""
    for cand in \
        "${RUN_DIR}/weights/best.pt" \
        "${DEFAULT_RUN_DIR}/weights/best.pt" \
        "${WEIGHT_FILE}" ; do
        if [[ -f "${cand}" ]]; then
            EVAL_CKPT="${cand}"
            break
        fi
    done

    if [[ -z "${EVAL_CKPT}" ]]; then
        echo "ERROR: no checkpoint found for ${VARIANT}. Train first or place one at ${WEIGHT_FILE}." >&2
        exit 1
    fi

    "${PYTHON}" "${SCRIPT_DIR}/run_evaluate.py" "${EVAL_CKPT}" \
        --base-dir "${DATA_AUG_DIR}" \
        --work-dir "${WORK_DIR}" \
        --variant "${VARIANT}" \
        "${PRETRAINED_FLAG[@]}"

    # ---- Pseudo-label retrain (only when MODE=pseudo) ---------------------
    if [[ "${MODE}" == "pseudo" ]]; then
        echo ">>> [${VARIANT}] Pseudo-label retrain (run_pseudo_label.py)"

        if [[ -d "${DEFAULT_PSEUDO_RUN_DIR}" && "${DEFAULT_PSEUDO_RUN_DIR}" != "${PSEUDO_RUN_DIR}" ]]; then
            rm -rf "${DEFAULT_PSEUDO_RUN_DIR}"
        fi

        "${PYTHON}" "${SCRIPT_DIR}/run_pseudo_label.py" "${EVAL_CKPT}" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant "${VARIANT}" \
            "${PRETRAINED_FLAG[@]}"

        if [[ -d "${DEFAULT_PSEUDO_RUN_DIR}" ]]; then
            rm -rf "${PSEUDO_RUN_DIR}"
            mv "${DEFAULT_PSEUDO_RUN_DIR}" "${PSEUDO_RUN_DIR}"
            echo ">>> [${VARIANT}] Saved pseudo run -> ${PSEUDO_RUN_DIR}"
        fi
    fi
}

for v in "${VARIANTS[@]}"; do
    run_variant "${v}"
done

echo "============================================================"
echo " change_architecture_model.sh finished (mode=${MODE}, variants=${VARIANTS[*]})"
echo "============================================================"
