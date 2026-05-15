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
# Output: runs/yolov12m_<variant>/weights/best.pt
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
RUNS_PARENT="${SCRIPT_DIR}/runs"
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
    local DEFAULT_RUN_DIR="${RUNS_PARENT}/yolov12m"
    local RUN_DIR="${RUNS_PARENT}/yolov12m_${VARIANT}"

    echo
    echo "--- VARIANT=${VARIANT}  work=${WORK_DIR}  out=${RUN_DIR} ---"

    # ---- Train ----------------------------------------------------------------
    if [[ "${MODE}" != "eval" ]]; then
        # Start clean so Ultralytics writes to the default name (yolov12m).
        [[ -d "${DEFAULT_RUN_DIR}" ]] && rm -rf "${DEFAULT_RUN_DIR}"

        "${PYTHON}" "${SCRIPT_DIR}/run_baseline_train.py" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant  "${VARIANT}"

        # Rename runs/yolov12m → runs/yolov12m_<variant>
        if [[ -d "${DEFAULT_RUN_DIR}" ]]; then
            rm -rf "${RUN_DIR}"
            mv "${DEFAULT_RUN_DIR}" "${RUN_DIR}"
            echo ">>> Saved → ${RUN_DIR}"
        fi
    fi

    # ---- Eval -----------------------------------------------------------------
    if [[ "${MODE}" != "train" ]]; then
        local CKPT=""
        for cand in "${RUN_DIR}/weights/best.pt" "${DEFAULT_RUN_DIR}/weights/best.pt"; do
            [[ -f "${cand}" ]] && { CKPT="${cand}"; break; }
        done

        if [[ -z "${CKPT}" ]]; then
            echo "ERROR: no baseline checkpoint found for variant '${VARIANT}'." >&2
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
