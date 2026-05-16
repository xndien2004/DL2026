"""Training entry — V4 3-phase SimAM training. Loads from baseline checkpoint."""
from __future__ import annotations
import os
from . import config as cfg
from core.data_preparation import build_dataset, split_dataframes
from core.detection_base import DetectionModel
from . import ultralytics_detector  # noqa: F401
from .simam import create_custom_model_yaml, inject_simam
from .ultralytics_detector import register_simam


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


def main(**overrides) -> None:
    cfg.apply_overrides(**overrides)
    cfg.print_config()

    df = build_dataset(
        base_dir=cfg.BASE_DIR, work_dir=cfg.WORK_DIR,
        data_variant=cfg.DATA_VARIANT, num_classes=cfg.NUM_CLASSES,
        splits=cfg.SPLITS,
    )
    split_dataframes(df)

    train_model_name = _setup_model_name()
    print(f"Training model: {train_model_name}")
    if cfg.PRETRAINED_CKPT:
        if os.path.isfile(cfg.PRETRAINED_CKPT):
            print(f"  Loading from baseline: {cfg.PRETRAINED_CKPT}")
        else:
            print(f"  WARNING: PRETRAINED_CKPT not found: {cfg.PRETRAINED_CKPT}")
            print(f"  Run run_baseline_train.py first to produce the baseline checkpoint.")

    model = DetectionModel.create(
        train_model_name, num_classes=cfg.NUM_CLASSES,
        class_names=cfg.CLASS_NAMES, device=cfg.DEVICE,
    )

    total = sum(p.numel() for p in model.model.parameters())
    print(f"  Total params: {total:,}")

    output_name = f"yolov12m_new_{cfg.DATA_VARIANT}"

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
            output_name=output_name,
        )

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
            output_name=output_name,
        )

    print("=" * 70)
    print("Phase 3: freeze=0, anti-overfit + cutmix (45 epochs)")
    print("=" * 70)
    model.train(
        epochs=45, lr=0.0001, base_dir=cfg.WORK_DIR,
        imgsz=768, batch=16, patience=cfg.PATIENCE,
        augmentation={**cfg.AUGMENTATION, "mixup": 0.20, "copy_paste": 0.35, "cutmix": 0.10},
        optimizer="SGD", close_mosaic=15, cos_lr=True,
        warmup_epochs=1, warmup_momentum=0.5,
        box=8.0, cls=2.5, dfl=2.0,
        label_smoothing=0.10, dropout=0.15,
        nbs=64, rect=False, freeze=0, save_period=5,
        output_name=output_name,
    )

    out_dir = cfg.WORK_ROOT / "output" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.plot_training_curves(save_path=str(out_dir / "training_curves.png"))


if __name__ == "__main__":
    main()
