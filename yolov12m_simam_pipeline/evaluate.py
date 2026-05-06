"""Evaluation entry — replicates V4 sections VII–IX."""

from __future__ import annotations

import os
from pathlib import Path

from . import config as cfg
from .data_preparation import build_dataset, split_dataframes
from .dataset import build_dataloaders
from .detection_base import DetectionModel
from . import ultralytics_detector  # noqa: F401
from .evaluation import export_predictions_csv
from .metrics import collect_predictions
from .plots import plot_confusion_matrix, plot_f1_curves, plot_pr_curves
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


def find_checkpoint(model_name: str) -> str:
    candidates = [
        str(cfg.WORK_ROOT / "runs" / model_name / "weights" / "best.pt"),
        str(cfg.WORK_ROOT / "runs" / f"{model_name}_pseudo" / "weights" / "best.pt"),
        cfg.PRETRAINED_CKPT,
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No checkpoint found. Tried: {candidates}\n"
        f"Pass a path explicitly via main(checkpoint=...) or as the first CLI arg."
    )


def main(checkpoint: str | None = None, **overrides) -> None:
    cfg.apply_overrides(**overrides)
    cfg.print_config()

    df = build_dataset(
        base_dir=cfg.BASE_DIR,
        work_dir=cfg.WORK_DIR,
        data_variant=cfg.DATA_VARIANT,
    )
    train_df, valid_df, test_df = split_dataframes(df)

    loaders = build_dataloaders(train_df, valid_df, test_df,
                                batch_size=cfg.BATCH_SIZE, work_dir=cfg.WORK_DIR)

    train_model_name = _setup_model_name()
    out_prefix = train_model_name

    model = DetectionModel.create(
        train_model_name,
        num_classes=cfg.NUM_CLASSES,
        class_names=cfg.CLASS_NAMES,
        device=cfg.DEVICE,
    )
    model.load(checkpoint or find_checkpoint(train_model_name))
    # Always re-create the data.yaml so it points at the local WORK_DIR.
    model.yolo_data_yaml = model._create_data_yaml(cfg.WORK_DIR)

    test_metrics = model.evaluate(
        iou_threshold=cfg.IOU_THRESHOLD,
        score_threshold=cfg.SCORE_THRESHOLD,
        base_dir=cfg.WORK_DIR,
        split="test",
    )
    model.print_metrics(test_metrics)
    model.plot_metrics_per_class(test_metrics, save_path=f"{out_prefix}_test_metrics.png")
    model.plot_training_curves(save_path=f"{out_prefix}_training_curves.png")

    # ---- Advanced metrics + plots --------------------------------------
    print("Collecting raw predictions...")
    dets, gts = collect_predictions(model, loaders["test_dataset"], score_threshold=0.01)
    print(f"Done — {len(dets)} images")

    plot_confusion_matrix(model, dets, gts,
                          score_thr=cfg.SCORE_THRESHOLD, iou_thr=cfg.IOU_THRESHOLD,
                          save_path=f"{out_prefix}_confusion_matrix.png")
    plot_pr_curves(model, dets, gts, test_metrics,
                   iou_thr=cfg.IOU_THRESHOLD,
                   save_path=f"{out_prefix}_pr_curves.png")
    plot_f1_curves(model, dets, gts,
                   iou_thr=cfg.IOU_THRESHOLD,
                   save_path=f"{out_prefix}_f1_curves.png")

    # ---- TP/FP/FN CSV --------------------------------------------------
    export_predictions_csv(
        out_prefix, dets, gts,
        image_ids=loaders["test_dataset"].image_ids,
        iou_thr=cfg.IOU_THRESHOLD, score_thr=cfg.SCORE_THRESHOLD,
        csv_out=f"{out_prefix}_test_predictions.csv",
    )

    # ---- Visualize predictions -----------------------------------------
    model.visualize_predictions(
        test_df, cfg.WORK_DIR,
        n_examples=cfg.N_EXAMPLES,
        save_dir=f"{out_prefix}_predictions",
        score_threshold=cfg.SCORE_THRESHOLD,
    )


if __name__ == "__main__":
    main()
