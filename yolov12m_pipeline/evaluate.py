"""Evaluation entry point — replicates Sections VII–IX of the notebook."""

from __future__ import annotations

import os

from . import config as cfg
from .data_preparation import build_dataset, split_dataframes
from .dataset import build_dataloaders
from .detection_base import DetectionModel
from . import ultralytics_detector  # noqa: F401
from .evaluation import export_predictions_csv
from .metrics import collect_predictions
from .plots import plot_confusion_matrix, plot_f1_curves, plot_pr_curves


DEFAULT_CHECKPOINT_CANDIDATES = [
    f"/kaggle/working/runs/detect/detection_runs/{cfg.MODEL_NAME}/weights/best.pt",
    f"detection_runs/{cfg.MODEL_NAME}/weights/best.pt",
]


def find_checkpoint(candidates=None) -> str:
    candidates = candidates or DEFAULT_CHECKPOINT_CANDIDATES
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No checkpoint found. Tried: {candidates}\n"
        f"Pass a path explicitly via main(checkpoint=...)"
    )


def main(checkpoint: str | None = None, *, prepare_data: bool = True, **overrides) -> None:
    cfg.apply_overrides(**overrides)
    cfg.print_config()

    if prepare_data:
        df = build_dataset(
            base_dir=cfg.BASE_DIR,
            work_dir=cfg.WORK_DIR,
            data_variant=cfg.DATA_VARIANT,
        )
        train_df, valid_df, test_df = split_dataframes(df)
    else:
        # Caller already prepared data; rebuild dataframes from disk would go here.
        raise NotImplementedError("Set prepare_data=True or wire up your own df loader.")

    loaders = build_dataloaders(train_df, valid_df, test_df,
                                batch_size=cfg.BATCH_SIZE, work_dir=cfg.WORK_DIR)

    model = DetectionModel.create(
        cfg.MODEL_NAME,
        num_classes=cfg.NUM_CLASSES,
        class_names=cfg.CLASS_NAMES,
        device=cfg.DEVICE,
    )
    model.load(checkpoint or find_checkpoint())

    test_metrics = model.evaluate(
        iou_threshold=cfg.IOU_THRESHOLD,
        score_threshold=cfg.SCORE_THRESHOLD,
        base_dir=cfg.WORK_DIR,
        split="test",
    )
    model.print_metrics(test_metrics)
    model.plot_metrics_per_class(test_metrics, save_path=f"{cfg.MODEL_NAME}_test_metrics.png")
    model.plot_training_curves(save_path=f"{cfg.MODEL_NAME}_training_curves.png")

    # ---- Advanced metrics + plots -------------------------------------------
    print("Collecting raw predictions...")
    dets, gts = collect_predictions(model, loaders["test_dataset"], score_threshold=0.01)
    print(f"Done — {len(dets)} images")

    plot_confusion_matrix(model, dets, gts,
                          score_thr=cfg.SCORE_THRESHOLD, iou_thr=cfg.IOU_THRESHOLD,
                          save_path=f"{cfg.MODEL_NAME}_confusion_matrix.png")
    plot_pr_curves(model, dets, gts, test_metrics,
                   iou_thr=cfg.IOU_THRESHOLD,
                   save_path=f"{cfg.MODEL_NAME}_pr_curves.png")
    plot_f1_curves(model, dets, gts,
                   iou_thr=cfg.IOU_THRESHOLD,
                   save_path=f"{cfg.MODEL_NAME}_f1_curves.png")

    # ---- TP/FP/FN CSV --------------------------------------------------------
    export_predictions_csv(
        cfg.MODEL_NAME, dets, gts,
        image_ids=loaders["test_dataset"].image_ids,
        iou_thr=cfg.IOU_THRESHOLD, score_thr=cfg.SCORE_THRESHOLD,
        csv_out=f"{cfg.MODEL_NAME}_test_predictions.csv",
    )

    # ---- Visualize predictions -----------------------------------------------
    model.visualize_predictions(
        test_df, cfg.WORK_DIR,
        n_examples=cfg.N_EXAMPLES,
        save_dir=f"{cfg.MODEL_NAME}_predictions",
        score_threshold=cfg.SCORE_THRESHOLD,
    )


if __name__ == "__main__":
    main()
