"""Centralized configuration for the YOLOv12m + SimAM (V4) pipeline."""

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

# Pretrained checkpoint to fine-tune from. Override via CLI or env var.
PRETRAINED_CKPT = str((WORK_ROOT / "runs" / "yolov12m" / "weights" / "best.pt").resolve())


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
DATA_VARIANT = "mix"  # one of: "mix", "bright", "dark"
CLASS_NAMES = ["Broken", "Chipped", "Scratched", "Severe_Rust", "Tip_Wear"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_NAME = {i + 1: name for i, name in enumerate(CLASS_NAMES)}

# V4 notebook uses 'val' (not 'valid') for the validation split.
SPLITS = ["train", "val", "test"]
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VARIANT_ALIASES = {
    "collect": "mix", "all": "mix", "mixed": "mix",
    "bright_field": "bright", "bf": "bright",
    "dark_field": "dark", "df": "dark",
}


# ---------------------------------------------------------------------------
# Custom architecture
# ---------------------------------------------------------------------------
CUSTOM_ARCH = "simam"  # set to "" / None to disable and run plain yolov12m


# ---------------------------------------------------------------------------
# Training hyperparameters (V4)
# ---------------------------------------------------------------------------
MODEL_NAME = "yolov12m"
NUM_EPOCHS = 75            # 3 phases (3 freeze + 15 head + 45 full = 63 + slack)
LEARNING_RATE = 0.001
IMGSZ = 640
BATCH_SIZE = 8             # PyTorch DataLoader (eval-side)
BATCH_SIZE_YOLO = 24       # Ultralytics trainer batch
PATIENCE = 15
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.15
N_EXAMPLES = 100

# If True: skip Phase 1+2 and only run Phase 3 polish (when starting from a
# pre-converged checkpoint such as a previous V4 best.pt).
SKIP_PHASE_1_2 = True

AUGMENTATION = {
    "hsv_h": 0.015, "hsv_s": 0.5, "hsv_v": 0.3,
    "degrees": 25.0,
    "translate": 0.1, "scale": 0.5,
    "shear": 0.0,
    "fliplr": 0.5, "flipud": 0.3,
    "mosaic": 1.0,
    "mixup": 0.10,
    "copy_paste": 0.2,
    "erasing": 0.05,
}


# ---------------------------------------------------------------------------
# Pseudo-label settings
# ---------------------------------------------------------------------------
PSEUDO_CONF = 0.85
PSEUDO_IOU_OVERLAP = 0.30


_PATH_FIELDS = {"BASE_DIR", "WORK_DIR", "WORK_ROOT"}


def apply_overrides(**kwargs) -> None:
    """Mutate module-level config attributes at runtime.

    Skips ``None`` values so an unset CLI flag leaves the default alone.
    Path-typed fields are coerced to ``pathlib.Path``.
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

    if "CLASS_NAMES" in kwargs:
        mod.NUM_CLASSES = len(mod.CLASS_NAMES)
        mod.LABEL_TO_NAME = {i + 1: name for i, name in enumerate(mod.CLASS_NAMES)}


def print_config() -> None:
    print(f"BASE_DIR={BASE_DIR}  WORK_DIR={WORK_DIR}  WORK_ROOT={WORK_ROOT}")
    print(f"MODEL={MODEL_NAME}  CUSTOM_ARCH={CUSTOM_ARCH}  PRETRAINED={PRETRAINED_CKPT}")
    print(f"VARIANT={DATA_VARIANT}  CLASSES={CLASS_NAMES}")
    print(f"EPOCHS={NUM_EPOCHS}  LR={LEARNING_RATE}  IMGSZ={IMGSZ}  "
          f"BATCH={BATCH_SIZE_YOLO}  PATIENCE={PATIENCE}")
    print(f"IOU_THR={IOU_THRESHOLD}  SCORE_THR={SCORE_THRESHOLD}  Device={DEVICE}")
