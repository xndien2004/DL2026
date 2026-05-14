"""Centralized configuration: paths, classes, hyperparameters, device."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Paths (default: current working directory)
# ---------------------------------------------------------------------------
WORK_ROOT = Path.cwd()
BASE_DIR = (WORK_ROOT / "data" / "DataAug").resolve()
WORK_DIR = WORK_ROOT / "yolo_dataset"


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
DATA_VARIANT = "mix"  # one of: "mix", "bright", "dark"
CLASS_NAMES = ["Broken", "Chipped", "Scratched", "Severe_Rust", "Tip_Wear"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_NAME = {i + 1: name for i, name in enumerate(CLASS_NAMES)}

SPLITS = ["train", "valid", "test"]
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VARIANT_ALIASES = {
    "collect": "mix", "all": "mix", "mixed": "mix",
    "bright_field": "bright", "bf": "bright",
    "dark_field": "dark", "df": "dark",
}


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME = "yolov12m"
NUM_EPOCHS = 500
LEARNING_RATE = 0.01
IMGSZ = 640
BATCH_SIZE = 8           # PyTorch DataLoader (eval-side)
BATCH_SIZE_YOLO = 256     # Ultralytics trainer batch
PATIENCE = 50
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.2
N_EXAMPLES = 100

AUGMENTATION = {
    "hsv_h": 0.015, "hsv_s": 0.5, "hsv_v": 0.3,
    "degrees": 15.0, "translate": 0.1, "scale": 0.3, "shear": 0.0,
    "fliplr": 0.5, "flipud": 0.3,
    "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.1, "erasing": 0.05,
}


_PATH_FIELDS = {"BASE_DIR", "WORK_DIR", "WORK_ROOT"}


def apply_overrides(**kwargs) -> None:
    """Mutate module-level config attributes at runtime.

    Skips ``None`` values (so an unset CLI flag leaves the default alone).
    Path-typed fields are coerced to ``pathlib.Path``. Raises ``ValueError``
    on unknown keys to catch typos early.
    """
    import sys

    mod = sys.modules[__name__]
    for key, val in kwargs.items():
        if val is None:
            continue
        if not hasattr(mod, key):
            raise ValueError(f"Unknown config field: {key!r}")
        if key in _PATH_FIELDS and not isinstance(val, Path):
            val = Path(val).resolve()
        setattr(mod, key, val)

    # Re-derive dependent fields if classes changed
    if "CLASS_NAMES" in kwargs:
        mod.NUM_CLASSES = len(mod.CLASS_NAMES)
        mod.LABEL_TO_NAME = {i + 1: name for i, name in enumerate(mod.CLASS_NAMES)}


def print_config() -> None:
    print(f"BASE_DIR={BASE_DIR}  WORK_DIR={WORK_DIR}  MODEL={MODEL_NAME}")
    print(f"VARIANT={DATA_VARIANT}  CLASSES={CLASS_NAMES}")
    print(f"EPOCHS={NUM_EPOCHS}  LR={LEARNING_RATE}  IMGSZ={IMGSZ}  "
          f"BATCH={BATCH_SIZE_YOLO}  PATIENCE={PATIENCE}")
    print(f"IOU_THR={IOU_THRESHOLD}  SCORE_THR={SCORE_THRESHOLD}  Device={DEVICE}")
