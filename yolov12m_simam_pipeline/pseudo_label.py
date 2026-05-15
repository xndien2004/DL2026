"""Pseudo-labeling self-training (V4 Section XI/XII).

1. Run a trained V4 checkpoint over the train images with TTA.
2. Append high-confidence predictions that don't overlap any GT to
   each label file -> ``yolo_dataset_pseudo/``.
3. Retrain V4 on the expanded set with stronger regularization.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
import yaml as _yaml
from PIL import Image
from tqdm.auto import tqdm

from . import config as cfg
from core.data_preparation import build_dataset
from .simam import create_custom_model_yaml, inject_simam
from .ultralytics_detector import register_simam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _box_iou_pair(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / max(union, 1e-6)


def _yolo_to_xyxy(line: str, w: int, h: int):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls, cx, cy, bw, bh = (int(parts[0]),) + tuple(float(p) for p in parts[1:5])
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return [cls, x1, y1, x2, y2]


def _xyxy_to_yolo(cls: int, x1, y1, x2, y2, w: int, h: int) -> str:
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _setup_model_name() -> str:
    if not cfg.CUSTOM_ARCH:
        return cfg.MODEL_NAME
    inject_simam()
    yaml_path = create_custom_model_yaml(
        cfg.CUSTOM_ARCH, cfg.NUM_CLASSES,
        base_size=cfg.MODEL_NAME[-1] if cfg.MODEL_NAME[-1] in "nsmlx" else "m",
        out_dir=cfg.WORK_ROOT,
    )
    return register_simam(cfg.MODEL_NAME, yaml_path, pretrained=cfg.PRETRAINED_CKPT)


def _find_checkpoint(train_model_name: str) -> str:
    candidates = [
        str(cfg.WORK_ROOT / "runs" / train_model_name / "weights" / "best.pt"),
        cfg.PRETRAINED_CKPT,
    ]
    for p in candidates:
        if p and Path(p).is_file():
            return p
    raise FileNotFoundError(f"V4 best.pt not found. Tried: {candidates}")


# ---------------------------------------------------------------------------
# Step 1: generate pseudo-labels
# ---------------------------------------------------------------------------
def generate_pseudo_labels(checkpoint: str, expanded_dir: Path) -> None:
    """Read GT labels + run TTA predictions; append non-overlapping high-conf
    detections to each train label file in ``expanded_dir``.
    """
    from ultralytics import YOLO

    print(f"Loading V4 from: {checkpoint}")
    v4_model = YOLO(checkpoint)

    if expanded_dir.exists():
        shutil.rmtree(expanded_dir)
    for split in ["train", "val", "test"]:
        (expanded_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (expanded_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    print(f"Expanded dir: {expanded_dir}")

    # Symlink val + test as-is
    for split in ["val", "test"]:
        src_img = cfg.WORK_DIR / split / "images"
        src_lbl = cfg.WORK_DIR / split / "labels"
        if src_img.is_dir():
            for f in src_img.iterdir():
                dst = expanded_dir / split / "images" / f.name
                try:
                    os.symlink(f, dst)
                except OSError:
                    shutil.copy(f, dst)
        if src_lbl.is_dir():
            for f in src_lbl.iterdir():
                shutil.copy(f, expanded_dir / split / "labels" / f.name)
        n = len(list((expanded_dir / split / "images").iterdir()))
        print(f"  Copied {split}: {n} images")

    train_imgs = sorted((cfg.WORK_DIR / "train" / "images").iterdir())
    print(f"\nGenerating pseudo-labels on {len(train_imgs)} train images...")

    stats = {"images": 0, "gt_boxes": 0, "pseudo_added": 0,
             "skipped_overlap": 0, "skipped_lowconf": 0}

    for img_path in tqdm(train_imgs, desc="Pseudo-labeling"):
        dst_img = expanded_dir / "train" / "images" / img_path.name
        try:
            os.symlink(img_path, dst_img)
        except OSError:
            shutil.copy(img_path, dst_img)

        src_lbl = cfg.WORK_DIR / "train" / "labels" / (img_path.stem + ".txt")
        gt_lines = [l for l in src_lbl.read_text().splitlines() if l.strip()] if src_lbl.exists() else []

        with Image.open(img_path) as im:
            w, h = im.size

        gt_boxes = [_yolo_to_xyxy(l, w, h) for l in gt_lines]
        gt_boxes = [b for b in gt_boxes if b is not None]
        stats["gt_boxes"] += len(gt_boxes)

        device = 0 if torch.cuda.is_available() else "cpu"
        res = v4_model.predict(str(img_path), conf=0.5, iou=0.7, verbose=False,
                               device=device, augment=True, imgsz=cfg.IMGSZ)[0]
        pseudo_extras = []
        if res.boxes is not None and len(res.boxes) > 0:
            for i in range(len(res.boxes)):
                conf = float(res.boxes.conf[i].cpu())
                pred_cls = int(res.boxes.cls[i].cpu())
                x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy()

                if conf < cfg.PSEUDO_CONF:
                    stats["skipped_lowconf"] += 1
                    continue

                overlap = False
                for gt_cls, gx1, gy1, gx2, gy2 in gt_boxes:
                    if gt_cls == pred_cls and _box_iou_pair(
                            [x1, y1, x2, y2], [gx1, gy1, gx2, gy2]) > cfg.PSEUDO_IOU_OVERLAP:
                        overlap = True
                        break
                if overlap:
                    stats["skipped_overlap"] += 1
                    continue

                pseudo_extras.append((pred_cls, x1, y1, x2, y2))
                stats["pseudo_added"] += 1

        dst_lbl = expanded_dir / "train" / "labels" / (img_path.stem + ".txt")
        with open(dst_lbl, "w") as f:
            for line in gt_lines:
                f.write(line + "\n")
            for c, x1, y1, x2, y2 in pseudo_extras:
                f.write(_xyxy_to_yolo(c, x1, y1, x2, y2, w, h) + "\n")
        stats["images"] += 1

    print(f"\n{'=' * 60}")
    print(f"PSEUDO-LABELING SUMMARY (conf >= {cfg.PSEUDO_CONF})")
    print(f"{'=' * 60}")
    print(f"  Train images processed:  {stats['images']}")
    print(f"  Original GT boxes:       {stats['gt_boxes']}")
    print(f"  Pseudo-labels added:     {stats['pseudo_added']} "
          f"(+{stats['pseudo_added'] / max(stats['gt_boxes'], 1) * 100:.1f}%)")
    print(f"  Skipped (overlap GT):    {stats['skipped_overlap']}")
    print(f"  Skipped (conf < {cfg.PSEUDO_CONF}): {stats['skipped_lowconf']}")
    print(f"\nExpanded dataset ready: {expanded_dir}")


# ---------------------------------------------------------------------------
# Step 2: retrain on expanded set
# ---------------------------------------------------------------------------
def retrain_on_expanded(checkpoint: str, expanded_dir: Path,
                        train_model_name: str, epochs: int = 30) -> Path:
    from ultralytics import YOLO

    expanded_yaml = cfg.WORK_ROOT / "yolo_data_pseudo.yaml"
    with open(expanded_yaml, "w") as f:
        _yaml.dump({
            "path":  str(expanded_dir),
            "train": "train/images",
            "val":   "val/images",
            "test":  "test/images",
            "nc":    cfg.NUM_CLASSES,
            "names": list(cfg.CLASS_NAMES),
        }, f)
    print(f"Created expanded data.yaml: {expanded_yaml}")

    safe_resume = cfg.WORK_ROOT / f"{train_model_name}_pseudo_base.pt"
    shutil.copy(checkpoint, safe_resume)
    print(f"Loaded base from {checkpoint} -> {safe_resume}")

    retrain_model = YOLO(str(safe_resume))

    print("=" * 70)
    print(f"Pseudo-label retrain: {epochs} epochs on expanded dataset")
    print("=" * 70)
    device = 0 if torch.cuda.device_count() == 1 else ("0,1" if torch.cuda.device_count() >= 2 else "cpu")
    retrain_model.train(
        data=str(expanded_yaml),
        epochs=epochs, lr0=0.00005,
        imgsz=768, batch=16,
        patience=10,
        optimizer="SGD", momentum=0.937,
        close_mosaic=10, cos_lr=True,
        warmup_epochs=1, warmup_momentum=0.5,
        box=8.0, cls=2.5, dfl=2.0,
        label_smoothing=0.15, dropout=0.20,
        nbs=64, rect=False, freeze=0,
        save_period=5,
        project=str(cfg.WORK_ROOT / "runs"),
        name=f"{train_model_name}_pseudo",
        exist_ok=True,
        device=device, workers=4,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        degrees=25.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.3,
        mosaic=1.0, mixup=0.20, copy_paste=0.35, cutmix=0.10, erasing=0.05,
    )

    # Eval afterwards
    final_path = cfg.WORK_ROOT / "runs" / f"{train_model_name}_pseudo" / "weights" / "best.pt"
    print("\n" + "=" * 70)
    print("Eval V4-pseudo on test set")
    print("=" * 70)
    if final_path.exists():
        final_model = YOLO(str(final_path))
        eval_yaml = cfg.WORK_ROOT / "yolo_data.yaml"
        if not eval_yaml.exists():
            eval_yaml = expanded_yaml
        metrics = final_model.val(
            data=str(eval_yaml), split="test",
            imgsz=768, batch=16, device=device,
            conf=0.15, iou=0.5, augment=True, verbose=True,
        )
        print(f"\n★ V4-PSEUDO Test mAP@0.5 = {metrics.box.map50:.4f}")
        print(f"   V4-PSEUDO Test mAP@0.5:0.95 = {metrics.box.map:.4f}")
    else:
        print(f"⚠ {final_path} not found — training may not have completed")
    return final_path


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------
def main(checkpoint: str | None = None, *, retrain_epochs: int = 30, **overrides) -> None:
    cfg.apply_overrides(**overrides)
    cfg.print_config()

    # Make sure base WORK_DIR is populated (won't re-download — only re-symlinks)
    build_dataset(
        base_dir=cfg.BASE_DIR,
        work_dir=cfg.WORK_DIR,
        data_variant=cfg.DATA_VARIANT,
        num_classes=cfg.NUM_CLASSES,
        splits=cfg.SPLITS,
    )

    train_model_name = _setup_model_name()
    ckpt = checkpoint or _find_checkpoint(train_model_name)
    expanded_dir = cfg.WORK_ROOT / "yolo_dataset_pseudo"

    generate_pseudo_labels(ckpt, expanded_dir)
    retrain_on_expanded(ckpt, expanded_dir, train_model_name, epochs=retrain_epochs)


if __name__ == "__main__":
    main()
