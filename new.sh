#!/usr/bin/env bash
# Train + evaluate YOLOv12m + SimAM.
# Run baseline.sh first to produce the pretrained checkpoint.
#
# Usage:
#   bash new.sh                    # train + eval, mix variant
#   bash new.sh full mix
#   bash new.sh full all           # iterate mix, bright, dark
#   bash new.sh train mix          # train only
#   bash new.sh eval  mix          # eval only (checkpoint must exist)
#
# Args:
#   $1 MODE     full | train | eval   (default: full)
#   $2 VARIANT  mix | bright | dark | all   (default: mix)
#
# Pretrained: auto-detected from output/yolov12m_base_<variant>/weights/best.pt
#             (produced by baseline.sh — run that first)
# Output:     output/yolov12m_new_<variant>/weights/best.pt
#
# Env:
#   PYTHON  Python executable (default: python3)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

MODE="${1:-full}"
VARIANT_ARG="${2:-mix}"

DATA_AUG_DIR="${SCRIPT_DIR}/data/DataAug"
PYTHON="${PYTHON:-python3}"

case "${VARIANT_ARG}" in
    mix|bright|dark) VARIANTS=("${VARIANT_ARG}") ;;
    all)             VARIANTS=("mix" "bright" "dark") ;;
    *) echo "ERROR: unknown variant '${VARIANT_ARG}'. Use mix|bright|dark|all." >&2; exit 2 ;;
esac

case "${MODE}" in
    full|train|eval) ;;
    *) echo "ERROR: unknown mode '${MODE}'. Use full|train|eval." >&2; exit 2 ;;
esac

echo "============================================================"
echo " [SimAM] ${SCRIPT_DIR}"
echo " Mode      : ${MODE}"
echo " Variant(s): ${VARIANTS[*]}"
echo " Data      : ${DATA_AUG_DIR}"
echo "============================================================"

run_variant() {
    local VARIANT="$1"
    local WORK_DIR="${SCRIPT_DIR}/yolo_dataset_${VARIANT}"
    local OUTPUT_DIR="${SCRIPT_DIR}/output/yolov12m_new_${VARIANT}"
    local BASELINE_CKPT="${SCRIPT_DIR}/output/yolov12m_base_${VARIANT}/weights/best.pt"

    echo
    echo "--- VARIANT=${VARIANT}  work=${WORK_DIR}  out=${OUTPUT_DIR} ---"

    # Resolve baseline pretrained checkpoint
    local PRETRAINED_FLAG=()
    if [[ -f "${BASELINE_CKPT}" ]]; then
        PRETRAINED_FLAG=(--pretrained "${BASELINE_CKPT}")
        echo ">>> Pretrained: ${BASELINE_CKPT}"
    else
        echo ">>> WARNING: baseline checkpoint not found at ${BASELINE_CKPT}"
        echo "             Run 'bash baseline.sh full ${VARIANT}' first for best results."
    fi

    # ---- Train ----------------------------------------------------------------
    if [[ "${MODE}" != "eval" ]]; then
        "${PYTHON}" "${SCRIPT_DIR}/run_train.py" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant  "${VARIANT}" \
            "${PRETRAINED_FLAG[@]}"
        echo ">>> Saved → ${OUTPUT_DIR}"
    fi

    # ---- Eval -----------------------------------------------------------------
    if [[ "${MODE}" != "train" ]]; then
        local CKPT="${OUTPUT_DIR}/weights/best.pt"

        if [[ ! -f "${CKPT}" ]]; then
            echo "ERROR: no SimAM checkpoint found at ${CKPT}" >&2
            echo "       Run 'bash new.sh train ${VARIANT}' first." >&2
            exit 1
        fi

        "${PYTHON}" "${SCRIPT_DIR}/run_evaluate.py" "${CKPT}" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant  "${VARIANT}" \
            "${PRETRAINED_FLAG[@]}"
    fi
}

for v in "${VARIANTS[@]}"; do run_variant "${v}"; done

echo
echo "============================================================"
echo " new.sh done  mode=${MODE}  variants=${VARIANTS[*]}"
echo "============================================================"
