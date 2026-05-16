"""Training entry point — baseline single-phase."""
from __future__ import annotations
from . import config as cfg
from core.data_preparation import build_dataset, split_dataframes
from core.detection_base import DetectionModel
from . import ultralytics_detector  # noqa: F401


def main(**overrides) -> None:
    cfg.apply_overrides(**overrides)
    cfg.print_config()

    df = build_dataset(
        base_dir=cfg.BASE_DIR, work_dir=cfg.WORK_DIR,
        data_variant=cfg.DATA_VARIANT, num_classes=cfg.NUM_CLASSES,
        splits=cfg.SPLITS,
    )
    split_dataframes(df)

    model = DetectionModel.create(
        cfg.MODEL_NAME, num_classes=cfg.NUM_CLASSES,
        class_names=cfg.CLASS_NAMES, device=cfg.DEVICE,
    )
    output_name = f"yolov12m_base_{cfg.DATA_VARIANT}"
    model.train(
        epochs=cfg.NUM_EPOCHS, lr=cfg.LEARNING_RATE,
        base_dir=cfg.WORK_DIR, imgsz=cfg.IMGSZ,
        batch=cfg.BATCH_SIZE_YOLO, patience=cfg.PATIENCE,
        augmentation=cfg.AUGMENTATION, optimizer="SGD",
        close_mosaic=5, cos_lr=True, warmup_epochs=5,
        warmup_momentum=0.5, box=7.5, cls=2.0, dfl=1.5,
        label_smoothing=0.05, nbs=64, rect=False,
        output_name=output_name,
    )


if __name__ == "__main__":
    main()
