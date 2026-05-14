#!/usr/bin/env bash
# End-to-end driver for the BASELINE YOLOv12m pipeline (no SimAM).
#
# Layout (all relative to this script's directory):
#   ./data/DataAug                    <- dataset (auto-downloaded from HuggingFace)
#   ./weight/best.pt                  <- optional pretrained checkpoint
#   ./yolo_dataset_<variant>          <- prepared YOLO-flat layout (per variant)
#   ./detection_runs/yolov12m_<variant>/weights/best.pt   <- training output
#
# Usage:
#   bash baseline.sh                          # full + mix
#   bash baseline.sh full mix
#   bash baseline.sh full bright
#   bash baseline.sh full dark
#   bash baseline.sh full all                 # iterate mix, bright, dark
#   bash baseline.sh skip-install bright      # full pipeline, skip pip install
#   bash baseline.sh eval-only dark           # only evaluate
#
# Args:
#   $1 MODE     full | skip-install | eval-only       (default: full)
#   $2 VARIANT  mix  | bright | dark | all            (default: mix)
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

MODE="${1:-full}"
VARIANT_ARG="${2:-mix}"

DATA_DIR="${SCRIPT_DIR}/data"
DATA_AUG_DIR="${DATA_DIR}/DataAug"
WEIGHT_DIR="${SCRIPT_DIR}/weight"
WEIGHT_FILE="${WEIGHT_DIR}/best.pt"
RUNS_PARENT="${SCRIPT_DIR}/detection_runs"
HF_REPO="${HF_REPO:-nhonhoccode/DetectDrill}"

PYTHON="${PYTHON:-python}"

case "${VARIANT_ARG}" in
    mix|bright|dark)  VARIANTS=("${VARIANT_ARG}") ;;
    all)              VARIANTS=("mix" "bright" "dark") ;;
    *) echo "ERROR: unknown variant '${VARIANT_ARG}'. Use mix|bright|dark|all." >&2; exit 2 ;;
esac

case "${MODE}" in
    full|skip-install|eval-only) ;;
    *) echo "ERROR: unknown mode '${MODE}'. Use full|skip-install|eval-only." >&2; exit 2 ;;
esac

echo "============================================================"
echo " [BASELINE] Working dir: ${SCRIPT_DIR}"
echo " Mode:        ${MODE}"
echo " Variant(s):  ${VARIANTS[*]}"
echo " Data:        ${DATA_AUG_DIR}"
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
# 2. Download + unzip dataset if missing
# ---------------------------------------------------------------------------
if [[ -d "${DATA_AUG_DIR}/train" ]]; then
    echo ">>> Dataset already present at ${DATA_AUG_DIR}"
else
    if [[ ! -d "${DATA_AUG_DIR}" ]]; then
        echo ">>> Downloading dataset from HuggingFace (${HF_REPO})"
        "${PYTHON}" "${SCRIPT_DIR}/download_data.py" --repo "${HF_REPO}" --target "${DATA_DIR}"
    fi

    SNAPSHOT_DIR="${DATA_DIR}/_snapshot"
    if [[ -d "${SNAPSHOT_DIR}" ]]; then
        while IFS= read -r -d '' zf; do
            dest="${zf%.zip}"
            if [[ ! -d "${dest}" ]]; then
                echo ">>> Unzipping $(basename "${zf}")"
                unzip -q "${zf}" -d "${dest}"
            fi
        done < <(find "${SNAPSHOT_DIR}" -maxdepth 1 -name "*.zip" -print0)
    fi
fi

mkdir -p "${WEIGHT_DIR}"

# ---------------------------------------------------------------------------
# Per-variant pipeline
# ---------------------------------------------------------------------------
run_variant() {
    local VARIANT="$1"
    local WORK_DIR="${SCRIPT_DIR}/yolo_dataset_${VARIANT}"
    local RUN_NAME="yolov12m_${VARIANT}"
    local RUN_DIR="${RUNS_PARENT}/${RUN_NAME}"
    local DEFAULT_RUN_DIR="${RUNS_PARENT}/yolov12m"

    echo
    echo "------------------------------------------------------------"
    echo " VARIANT = ${VARIANT}"
    echo " work_dir = ${WORK_DIR}"
    echo " run_dir  = ${RUN_DIR}"
    echo "------------------------------------------------------------"

    # ---- Train ------------------------------------------------------------
    if [[ "${MODE}" != "eval-only" ]]; then
        # Ultralytics writes to detection_runs/yolov12m. Move any prior result
        # so this variant starts clean and we can rename afterwards.
        if [[ -d "${DEFAULT_RUN_DIR}" && "${DEFAULT_RUN_DIR}" != "${RUN_DIR}" ]]; then
            rm -rf "${DEFAULT_RUN_DIR}"
        fi

        echo ">>> [${VARIANT}] Training baseline (run_baseline_train.py)"
        "${PYTHON}" "${SCRIPT_DIR}/run_baseline_train.py" \
            --base-dir "${DATA_AUG_DIR}" \
            --work-dir "${WORK_DIR}" \
            --variant "${VARIANT}"

        # Rename detection_runs/yolov12m -> detection_runs/yolov12m_<variant>
        if [[ -d "${DEFAULT_RUN_DIR}" ]]; then
            rm -rf "${RUN_DIR}"
            mv "${DEFAULT_RUN_DIR}" "${RUN_DIR}"
            echo ">>> [${VARIANT}] Saved run -> ${RUN_DIR}"
        fi
    fi

    # ---- Evaluate ---------------------------------------------------------
    echo ">>> [${VARIANT}] Evaluating baseline (run_baseline_evaluate.py)"
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
        echo "ERROR: no baseline checkpoint found for ${VARIANT}. Train first or place one at ${WEIGHT_FILE}." >&2
        exit 1
    fi

    "${PYTHON}" "${SCRIPT_DIR}/run_baseline_evaluate.py" "${EVAL_CKPT}" \
        --base-dir "${DATA_AUG_DIR}" \
        --work-dir "${WORK_DIR}" \
        --variant "${VARIANT}"
}

for v in "${VARIANTS[@]}"; do
    run_variant "${v}"
done

echo "============================================================"
echo " baseline.sh finished (mode=${MODE}, variants=${VARIANTS[*]})"
echo "============================================================"
