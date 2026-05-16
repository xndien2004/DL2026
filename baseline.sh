#!/usr/bin/env bash
# Train + evaluate the baseline YOLOv12m pipeline.
# Run download.sh first to set up deps and data.
#
# Usage:
#   bash baseline.sh                    # train + eval, mix variant
#   bash baseline.sh full mix
#   bash baseline.sh full bright
#   bash baseline.sh full dark
#   bash baseline.sh full all           # iterate mix, bright, dark
#   bash baseline.sh train mix          # train only
#   bash baseline.sh eval  mix          # eval only (checkpoint must exist)
#
# Args:
#   $1 MODE     full | train | eval       (default: full)
#   $2 VARIANT  mix | bright | dark | all  (default: mix)
#
# Output: output/yolov12m_base_<variant>/weights/best.pt
#         (SimAM pipeline reads this as its pretrained checkpoint)
#
# Env:
#   PYTHON  Python executable (default: python3)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

MODE="${1:-full}"
VARIANT_ARG="${2:-mix}"

DATA_AUG_DIR="${SCRIPT_DIR}/data/DataAug/DataDrillDetect/DataAug"
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
echo " [BASELINE] ${SCRIPT_DIR}"
echo " Mode      : ${MODE}"
echo " Variant(s): ${VARIANTS[*]}"
echo " Data      : ${DATA_AUG_DIR}"
echo "============================================================"

run_variant() {
    local VARIANT="$1"
    local WORK_DIR="${SCRIPT_DIR}/yolo_dataset_${VARIANT}"
    local OUTPUT_DIR="${SCRIPT_DIR}/output/yolov12m_base_${VARIANT}"

    echo
    echo "--- VARIANT=${VARIANT}  work=${WORK_DIR}  out=${OUTPUT_DIR} ---"

    # ---- Train ----------------------------------------------------------------
    if [[ "${MODE}" != "eval" ]]; then
        "${PYTHON}" "${SCRIPT_DIR}/run_baseline_train.py" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant  "${VARIANT}"
        echo ">>> Saved → ${OUTPUT_DIR}"
    fi

    # ---- Eval -----------------------------------------------------------------
    if [[ "${MODE}" != "train" ]]; then
        local CKPT="${OUTPUT_DIR}/weights/best.pt"

        if [[ ! -f "${CKPT}" ]]; then
            echo "ERROR: no baseline checkpoint found at ${CKPT}" >&2
            echo "       Run 'bash baseline.sh train ${VARIANT}' first." >&2
            exit 1
        fi

        "${PYTHON}" "${SCRIPT_DIR}/run_baseline_evaluate.py" "${CKPT}" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant  "${VARIANT}"
    fi
}

for v in "${VARIANTS[@]}"; do run_variant "${v}"; done

echo
echo "============================================================"
echo " baseline.sh done  mode=${MODE}  variants=${VARIANTS[*]}"
echo "============================================================"
