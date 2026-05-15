"""CLI for the SimAM pipeline."""
from __future__ import annotations
import argparse
from core.cli import _add_common, _to_overrides

_TRAIN_FLAGS = {
    "model": "MODEL_NAME", "epochs": "NUM_EPOCHS", "lr": "LEARNING_RATE",
    "imgsz": "IMGSZ", "batch": "BATCH_SIZE_YOLO", "loader_batch": "BATCH_SIZE",
    "patience": "PATIENCE", "base_dir": "BASE_DIR", "work_dir": "WORK_DIR",
    "variant": "DATA_VARIANT", "iou": "IOU_THRESHOLD", "score": "SCORE_THRESHOLD",
    "pretrained": "PRETRAINED_CKPT", "custom_arch": "CUSTOM_ARCH",
    "skip_phase12": "SKIP_PHASE_1_2",
}
_EVAL_FLAGS = {
    "model": "MODEL_NAME", "base_dir": "BASE_DIR", "work_dir": "WORK_DIR",
    "variant": "DATA_VARIANT", "iou": "IOU_THRESHOLD", "score": "SCORE_THRESHOLD",
    "imgsz": "IMGSZ", "batch": "BATCH_SIZE_YOLO", "n_examples": "N_EXAMPLES",
    "custom_arch": "CUSTOM_ARCH", "pretrained": "PRETRAINED_CKPT",
}
_PSEUDO_FLAGS = {
    "model": "MODEL_NAME", "epochs": "NUM_EPOCHS", "imgsz": "IMGSZ",
    "batch": "BATCH_SIZE_YOLO", "patience": "PATIENCE",
    "base_dir": "BASE_DIR", "work_dir": "WORK_DIR", "variant": "DATA_VARIANT",
    "pretrained": "PRETRAINED_CKPT", "custom_arch": "CUSTOM_ARCH",
    "pseudo_conf": "PSEUDO_CONF", "pseudo_iou": "PSEUDO_IOU_OVERLAP",
}


def parse_train_args(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Train YOLOv12m + SimAM (V4).")
    _add_common(parser, _TRAIN_FLAGS)
    return _to_overrides(parser.parse_args(argv), _TRAIN_FLAGS)


def parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate SimAM checkpoint.")
    parser.add_argument("checkpoint", nargs="?", default=None)
    _add_common(parser, _EVAL_FLAGS)
    args = parser.parse_args(argv)
    return args.checkpoint, _to_overrides(args, _EVAL_FLAGS)


def parse_pseudo_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate pseudo-labels and retrain.")
    parser.add_argument("checkpoint", nargs="?", default=None)
    _add_common(parser, _PSEUDO_FLAGS)
    args = parser.parse_args(argv)
    return args.checkpoint, _to_overrides(args, _PSEUDO_FLAGS)
