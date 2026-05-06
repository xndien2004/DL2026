"""Training entry — replicates the V4 notebook's 3-phase Section VI."""

from __future__ import annotations

from . import config as cfg
from .data_preparation import build_dataset, split_dataframes
from .detection_base import DetectionModel
from . import ultralytics_detector  # noqa: F401  (registers ultralytics models)
from .simam import create_custom_model_yaml, inject_simam
from .ultralytics_detector import register_simam


def _setup_model_name() -> str:
    """Build SimAM YAML, monkey-patch ultralytics, register the variant."""
    if not cfg.CUSTOM_ARCH:
        return cfg.MODEL_NAME

    inject_simam()
    yaml_path = create_custom_model_yaml(
        cfg.CUSTOM_ARCH, cfg.NUM_CLASSES,
        base_size=cfg.MODEL_NAME[-1] if cfg.MODEL_NAME[-1] in "nsmlx" else "m",
        out_dir=cfg.WORK_ROOT,
    )
    return register_simam(cfg.MODEL_NAME, yaml_path, pretrained=cfg.PRETRAINED_CKPT)


def main(**overrides) -> None:
    cfg.apply_overrides(**overrides)
    cfg.print_config()

    df = build_dataset(
        base_dir=cfg.BASE_DIR,
        work_dir=cfg.WORK_DIR,
        data_variant=cfg.DATA_VARIANT,
    )
    split_dataframes(df)

    train_model_name = _setup_model_name()
    print(f"Training model: {train_model_name}")

    model = DetectionModel.create(
        train_model_name,
        num_classes=cfg.NUM_CLASSES,
        class_names=cfg.CLASS_NAMES,
        device=cfg.DEVICE,
    )

    total = sum(p.numel() for p in model.model.parameters())
    print(f"  Total params: {total:,}")

    # ---- Phase 1 — freeze=21 (lock everything but Detect head + SimAM) ----
    if not cfg.SKIP_PHASE_1_2:
        print("=" * 70)
        print("Phase 1: freeze=21, train Detect head + SimAM (3 epochs)")
        print("=" * 70)
        model.train(
            epochs=3, lr=0.001, base_dir=cfg.WORK_DIR,
            imgsz=cfg.IMGSZ, batch=cfg.BATCH_SIZE_YOLO, patience=10,
            augmentation={**cfg.AUGMENTATION, "mixup": 0.0, "copy_paste": 0.0},
            optimizer="AdamW", close_mosaic=2, cos_lr=False,
            warmup_epochs=1, warmup_momentum=0.8,
            box=7.5, cls=2.0, dfl=1.5,
            label_smoothing=0.05, nbs=64, rect=False,
            freeze=21, save_period=2,
        )

    # ---- Phase 2 — freeze=10 (unlock head, lock backbone) ----
    if not cfg.SKIP_PHASE_1_2:
        print("=" * 70)
        print("Phase 2: freeze=10, train head + SimAM + Detect (15 epochs)")
        print("=" * 70)
        model.train(
            epochs=15, lr=0.0005, base_dir=cfg.WORK_DIR,
            imgsz=cfg.IMGSZ, batch=cfg.BATCH_SIZE_YOLO, patience=10,
            augmentation=cfg.AUGMENTATION,
            optimizer="SGD", close_mosaic=5, cos_lr=True,
            warmup_epochs=1, warmup_momentum=0.5,
            box=7.5, cls=2.0, dfl=1.5,
            label_smoothing=0.05, nbs=64, rect=False,
            freeze=10, save_period=3,
        )

    # ---- Phase 3 — full unfreeze, anti-overfit (45 epochs) ----
    print("=" * 70)
    print("Phase 3: freeze=0, anti-overfit + cutmix (45 epochs)")
    print("=" * 70)
    model.train(
        epochs=45, lr=0.0001, base_dir=cfg.WORK_DIR,
        imgsz=768,
        batch=16,
        patience=cfg.PATIENCE,
        augmentation={
            **cfg.AUGMENTATION,
            "mixup": 0.20,
            "copy_paste": 0.35,
            "cutmix": 0.10,
        },
        optimizer="SGD", close_mosaic=15, cos_lr=True,
        warmup_epochs=1, warmup_momentum=0.5,
        box=8.0, cls=2.5, dfl=2.0,
        label_smoothing=0.10, dropout=0.15,
        nbs=64, rect=False, freeze=0, save_period=5,
    )


if __name__ == "__main__":
    main()
