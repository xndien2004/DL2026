"""argparse helpers shared by the run_*.py entry scripts.

Each CLI flag maps 1:1 to a config attribute. Unset flags leave the config
default in place (default=None + apply_overrides skips None values).
"""

from __future__ import annotations

import argparse


_TRAIN_FLAGS = {
    "model":         "MODEL_NAME",
    "epochs":        "NUM_EPOCHS",
    "lr":            "LEARNING_RATE",
    "imgsz":         "IMGSZ",
    "batch":         "BATCH_SIZE_YOLO",
    "loader_batch":  "BATCH_SIZE",
    "patience":      "PATIENCE",
    "base_dir":      "BASE_DIR",
    "work_dir":      "WORK_DIR",
    "variant":       "DATA_VARIANT",
    "iou":           "IOU_THRESHOLD",
    "score":         "SCORE_THRESHOLD",
    "pretrained":    "PRETRAINED_CKPT",
    "custom_arch":   "CUSTOM_ARCH",
    "skip_phase12":  "SKIP_PHASE_1_2",
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
    "custom_arch":  "CUSTOM_ARCH",
    "pretrained":   "PRETRAINED_CKPT",
}

_PSEUDO_FLAGS = {
    "model":         "MODEL_NAME",
    "epochs":        "NUM_EPOCHS",
    "imgsz":         "IMGSZ",
    "batch":         "BATCH_SIZE_YOLO",
    "patience":      "PATIENCE",
    "base_dir":      "BASE_DIR",
    "work_dir":      "WORK_DIR",
    "variant":       "DATA_VARIANT",
    "pretrained":    "PRETRAINED_CKPT",
    "custom_arch":   "CUSTOM_ARCH",
    "pseudo_conf":   "PSEUDO_CONF",
    "pseudo_iou":    "PSEUDO_IOU_OVERLAP",
}


def _add_common(parser: argparse.ArgumentParser, flags: dict[str, str]) -> None:
    add = parser.add_argument

    if "model" in flags:        add("--model",        type=str,   help="Model name (yolov12m, yolov8s, rtdetr-l, ...)")
    if "epochs" in flags:       add("--epochs",       type=int,   help="Number of training epochs")
    if "lr" in flags:           add("--lr",           type=float, help="Initial learning rate")
    if "imgsz" in flags:        add("--imgsz",        type=int,   help="Training/eval image size")
    if "batch" in flags:        add("--batch",        type=int,   help="Ultralytics trainer batch size")
    if "loader_batch" in flags: add("--loader-batch", dest="loader_batch", type=int,
                                    help="PyTorch DataLoader batch size (eval-side)")
    if "patience" in flags:     add("--patience",     type=int,   help="Early-stopping patience (epochs)")
    if "base_dir" in flags:     add("--base-dir",     dest="base_dir", type=str,
                                    help="Source dataset root (BASE_DIR)")
    if "work_dir" in flags:     add("--work-dir",     dest="work_dir", type=str,
                                    help="YOLO-flat working dir (WORK_DIR)")
    if "variant" in flags:      add("--variant",      type=str, choices=["mix", "bright", "dark"],
                                    help="Lighting subset")
    if "iou" in flags:          add("--iou",          type=float, help="IoU threshold")
    if "score" in flags:        add("--score",        type=float, help="Score threshold")
    if "n_examples" in flags:   add("--n-examples",   dest="n_examples", type=int,
                                    help="Number of test images to render with predictions")
    if "pretrained" in flags:   add("--pretrained",   type=str,
                                    help="Pretrained .pt checkpoint to fine-tune from")
    if "custom_arch" in flags:  add("--custom-arch",  dest="custom_arch", type=str,
                                    choices=["", "simam"], help="Custom architecture variant ('simam' or empty)")
    if "skip_phase12" in flags: add("--skip-phase12", dest="skip_phase12", action="store_true", default=None,
                                    help="Skip Phase 1+2 and run only Phase 3 polish")
    if "pseudo_conf" in flags:  add("--pseudo-conf",  dest="pseudo_conf", type=float,
                                    help="Pseudo-label confidence threshold")
    if "pseudo_iou" in flags:   add("--pseudo-iou",   dest="pseudo_iou", type=float,
                                    help="Pseudo-label IoU overlap threshold (skip if matches GT)")


def _to_overrides(args: argparse.Namespace, flags: dict[str, str]) -> dict:
    out: dict = {}
    for cli_name, cfg_name in flags.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            out[cfg_name] = val
    return out


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def parse_train_args(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Train YOLOv12m + SimAM (V4). Flags override the config module.",
    )
    _add_common(parser, _TRAIN_FLAGS)
    args = parser.parse_args(argv)
    return _to_overrides(args, _TRAIN_FLAGS)


def parse_eval_args(argv: list[str] | None = None) -> tuple[str | None, dict]:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint. Flags override the config module.",
    )
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to .pt checkpoint (default: auto-discover)")
    _add_common(parser, _EVAL_FLAGS)
    args = parser.parse_args(argv)
    return args.checkpoint, _to_overrides(args, _EVAL_FLAGS)


def parse_pseudo_args(argv: list[str] | None = None) -> tuple[str | None, dict]:
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels and retrain on expanded set.",
    )
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Trained .pt checkpoint to use as labeler (default: auto-discover)")
    _add_common(parser, _PSEUDO_FLAGS)
    args = parser.parse_args(argv)
    return args.checkpoint, _to_overrides(args, _PSEUDO_FLAGS)
