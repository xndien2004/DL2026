"""Training entry point — replicates the notebook's Section VI."""

from __future__ import annotations

from . import config as cfg
from .data_preparation import build_dataset, split_dataframes
from .detection_base import DetectionModel
from . import ultralytics_detector  # noqa: F401  (registers ultralytics models)


def main(**overrides) -> None:
    cfg.apply_overrides(**overrides)
    cfg.print_config()

    df = build_dataset(
        base_dir=cfg.BASE_DIR,
        work_dir=cfg.WORK_DIR,
        data_variant=cfg.DATA_VARIANT,
    )
    split_dataframes(df)

    model = DetectionModel.create(
        cfg.MODEL_NAME,
        num_classes=cfg.NUM_CLASSES,
        class_names=cfg.CLASS_NAMES,
        device=cfg.DEVICE,
    )

    model.train(
        epochs=cfg.NUM_EPOCHS,
        lr=cfg.LEARNING_RATE,
        base_dir=cfg.WORK_DIR,
        imgsz=cfg.IMGSZ,
        batch=cfg.BATCH_SIZE_YOLO,
        patience=cfg.PATIENCE,
        augmentation=cfg.AUGMENTATION,
        optimizer="SGD",
        close_mosaic=5,
        cos_lr=True,
        warmup_epochs=5,
        warmup_momentum=0.5,
        box=7.5,
        cls=2.0,
        dfl=1.5,
        label_smoothing=0.05,
        nbs=64,
        rect=False,
    )


if __name__ == "__main__":
    main()
