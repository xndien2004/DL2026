#!/usr/bin/env bash
# Train + evaluate YOLOv12m + SimAM.
# Run baseline.sh first to produce the pretrained checkpoint.
#
# Usage:
#   bash simam.sh                    # train + eval, mix variant
#   bash simam.sh full mix
#   bash simam.sh full all           # iterate mix, bright, dark
#   bash simam.sh train mix          # train only
#   bash simam.sh eval  mix          # eval only (checkpoint must exist)
#   bash simam.sh pseudo mix         # train + eval + pseudo-label retrain
#
# Args:
#   $1 MODE     full | train | eval | pseudo   (default: full)
#   $2 VARIANT  mix | bright | dark | all       (default: mix)
#
# Pretrained: auto-detected from runs/yolov12m_<variant>/weights/best.pt
#             (produced by baseline.sh — run that first)
# Output:     runs/yolov12m-simam_<variant>/weights/best.pt
#
# Env:
#   PYTHON  Python executable (default: python3)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

MODE="${1:-full}"
VARIANT_ARG="${2:-mix}"

DATA_AUG_DIR="${SCRIPT_DIR}/data/DataAug"
RUNS_PARENT="${SCRIPT_DIR}/runs"
PYTHON="${PYTHON:-python3}"

case "${VARIANT_ARG}" in
    mix|bright|dark) VARIANTS=("${VARIANT_ARG}") ;;
    all)             VARIANTS=("mix" "bright" "dark") ;;
    *) echo "ERROR: unknown variant '${VARIANT_ARG}'. Use mix|bright|dark|all." >&2; exit 2 ;;
esac

case "${MODE}" in
    full|train|eval|pseudo) ;;
    *) echo "ERROR: unknown mode '${MODE}'. Use full|train|eval|pseudo." >&2; exit 2 ;;
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
    local BASELINE_CKPT="${RUNS_PARENT}/yolov12m_${VARIANT}/weights/best.pt"
    local DEFAULT_RUN_DIR="${RUNS_PARENT}/yolov12m-simam"
    local RUN_DIR="${RUNS_PARENT}/yolov12m-simam_${VARIANT}"
    local DEFAULT_PSEUDO_DIR="${RUNS_PARENT}/yolov12m-simam_pseudo"
    local PSEUDO_DIR="${RUNS_PARENT}/yolov12m-simam_${VARIANT}_pseudo"

    echo
    echo "--- VARIANT=${VARIANT}  work=${WORK_DIR}  out=${RUN_DIR} ---"

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
        [[ -d "${DEFAULT_RUN_DIR}" ]] && rm -rf "${DEFAULT_RUN_DIR}"

        "${PYTHON}" "${SCRIPT_DIR}/run_train.py" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant  "${VARIANT}" \
            "${PRETRAINED_FLAG[@]}"

        # Rename runs/yolov12m-simam → runs/yolov12m-simam_<variant>
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
            echo "ERROR: no SimAM checkpoint found for variant '${VARIANT}'." >&2
            echo "       Run 'bash simam.sh train ${VARIANT}' first." >&2
            exit 1
        fi

        "${PYTHON}" "${SCRIPT_DIR}/run_evaluate.py" "${CKPT}" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant  "${VARIANT}" \
            "${PRETRAINED_FLAG[@]}"
    fi

    # ---- Pseudo-label retrain (pseudo mode only) ------------------------------
    if [[ "${MODE}" == "pseudo" ]]; then
        local CKPT=""
        for cand in "${RUN_DIR}/weights/best.pt" "${DEFAULT_RUN_DIR}/weights/best.pt"; do
            [[ -f "${cand}" ]] && { CKPT="${cand}"; break; }
        done

        if [[ -z "${CKPT}" ]]; then
            echo "ERROR: need SimAM checkpoint for pseudo-label step." >&2
            exit 1
        fi

        echo ">>> Pseudo-label retrain"
        [[ -d "${DEFAULT_PSEUDO_DIR}" ]] && rm -rf "${DEFAULT_PSEUDO_DIR}"

        "${PYTHON}" "${SCRIPT_DIR}/run_pseudo_label.py" "${CKPT}" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant  "${VARIANT}" \
            "${PRETRAINED_FLAG[@]}"

        if [[ -d "${DEFAULT_PSEUDO_DIR}" ]]; then
            rm -rf "${PSEUDO_DIR}"
            mv "${DEFAULT_PSEUDO_DIR}" "${PSEUDO_DIR}"
            echo ">>> Saved pseudo → ${PSEUDO_DIR}"
        fi
    fi
}

for v in "${VARIANTS[@]}"; do run_variant "${v}"; done

echo
echo "============================================================"
echo " simam.sh done  mode=${MODE}  variants=${VARIANTS[*]}"
echo "============================================================"
