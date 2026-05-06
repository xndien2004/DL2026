"""argparse helpers shared by the run_*.py entry scripts.

Each CLI flag maps 1:1 to a config attribute. Unset flags leave the
config default in place (we use ``default=None`` + ``apply_overrides``
to skip ``None`` values).
"""

from __future__ import annotations

import argparse


# Map CLI flag name -> config attribute name
_TRAIN_FLAGS = {
    "model":        "MODEL_NAME",
    "epochs":       "NUM_EPOCHS",
    "lr":           "LEARNING_RATE",
    "imgsz":        "IMGSZ",
    "batch":        "BATCH_SIZE_YOLO",
    "loader_batch": "BATCH_SIZE",
    "patience":     "PATIENCE",
    "base_dir":     "BASE_DIR",
    "work_dir":     "WORK_DIR",
    "variant":      "DATA_VARIANT",
    "iou":          "IOU_THRESHOLD",
    "score":        "SCORE_THRESHOLD",
}

_EVAL_FLAGS = {
    "model":        "MODEL_NAME",
    "base_dir":     "BASE_DIR",
    "work_dir":     "WORK_DIR",
    "variant":      "DATA_VARIANT",
    "iou":          "IOU_THRESHOLD",
    "score":        "SCORE_THRESHOLD",
    "imgsz":        "IMGSZ",
    "batch":        "BATCH_SIZE_YOLO",
    "n_examples":   "N_EXAMPLES",
}


def _add_common(parser: argparse.ArgumentParser, flags: dict[str, str]) -> None:
    """Register flags. All optional, default=None -> 'leave config alone'."""
    add = parser.add_argument

    if "model" in flags:        add("--model",      type=str,   help="Model name (e.g. yolov12m, yolov8s, rtdetr-l)")
    if "epochs" in flags:       add("--epochs",     type=int,   help="Number of training epochs")
    if "lr" in flags:           add("--lr",         type=float, help="Initial learning rate")
    if "imgsz" in flags:        add("--imgsz",      type=int,   help="Training/eval image size")
    if "batch" in flags:        add("--batch",      type=int,   help="Ultralytics trainer batch size")
    if "loader_batch" in flags: add("--loader-batch", dest="loader_batch", type=int,
                                    help="PyTorch DataLoader batch size (eval-side)")
    if "patience" in flags:     add("--patience",   type=int,   help="Early-stopping patience (epochs)")
    if "base_dir" in flags:     add("--base-dir",   dest="base_dir", type=str,
                                    help="Source dataset root (BASE_DIR)")
    if "work_dir" in flags:     add("--work-dir",   dest="work_dir", type=str,
                                    help="Working dir for prepared YOLO layout (WORK_DIR)")
    if "variant" in flags:      add("--variant",    type=str, choices=["mix", "bright", "dark"],
                                    help="Lighting subset")
    if "iou" in flags:          add("--iou",        type=float, help="IoU threshold for matching")
    if "score" in flags:        add("--score",      type=float, help="Score threshold for keeping detections")
    if "n_examples" in flags:   add("--n-examples", dest="n_examples", type=int,
                                    help="Number of test images to render with predictions")


def _to_overrides(args: argparse.Namespace, flags: dict[str, str]) -> dict:
    """Translate parsed flags -> {CONFIG_KEY: value} dict, skipping unset (None)."""
    out: dict = {}
    for cli_name, cfg_name in flags.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            out[cfg_name] = val
    return out


# ---------------------------------------------------------------------------
# Public builders used by run_*.py
# ---------------------------------------------------------------------------
def parse_train_args(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Train a detection model. All flags override yolov12m_pipeline.config.",
    )
    _add_common(parser, _TRAIN_FLAGS)
    args = parser.parse_args(argv)
    return _to_overrides(args, _TRAIN_FLAGS)


def parse_eval_args(argv: list[str] | None = None) -> tuple[str | None, dict]:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint. All flags override yolov12m_pipeline.config.",
    )
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to .pt checkpoint (default: auto-discover)")
    _add_common(parser, _EVAL_FLAGS)
    args = parser.parse_args(argv)
    return args.checkpoint, _to_overrides(args, _EVAL_FLAGS)
