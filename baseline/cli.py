"""CLI for the baseline pipeline. Flags map 1:1 to config attributes."""
from __future__ import annotations
import argparse
from core.cli import _add_common, _to_overrides

_TRAIN_FLAGS = {
    "model": "MODEL_NAME", "epochs": "NUM_EPOCHS", "lr": "LEARNING_RATE",
    "imgsz": "IMGSZ", "batch": "BATCH_SIZE_YOLO", "loader_batch": "BATCH_SIZE",
    "patience": "PATIENCE", "base_dir": "BASE_DIR", "work_dir": "WORK_DIR",
    "variant": "DATA_VARIANT", "iou": "IOU_THRESHOLD", "score": "SCORE_THRESHOLD",
}
_EVAL_FLAGS = {
    "model": "MODEL_NAME", "base_dir": "BASE_DIR", "work_dir": "WORK_DIR",
    "variant": "DATA_VARIANT", "iou": "IOU_THRESHOLD", "score": "SCORE_THRESHOLD",
    "imgsz": "IMGSZ", "batch": "BATCH_SIZE_YOLO", "n_examples": "N_EXAMPLES",
}


def parse_train_args(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Train baseline YOLOv12m.")
    _add_common(parser, _TRAIN_FLAGS)
    return _to_overrides(parser.parse_args(argv), _TRAIN_FLAGS)


def parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate baseline checkpoint.")
    parser.add_argument("checkpoint", nargs="?", default=None)
    _add_common(parser, _EVAL_FLAGS)
    args = parser.parse_args(argv)
    return args.checkpoint, _to_overrides(args, _EVAL_FLAGS)
