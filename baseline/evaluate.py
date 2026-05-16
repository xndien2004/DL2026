"""Evaluation entry point — baseline."""
from __future__ import annotations
import os
from pathlib import Path
from . import config as cfg
from core.data_preparation import build_dataset, split_dataframes
from core.dataset import build_dataloaders
from core.detection_base import DetectionModel
from core.evaluation import export_predictions_csv
from core.metrics import collect_predictions
from core.plots import plot_confusion_matrix, plot_f1_curves, plot_pr_curves
from . import ultralytics_detector  # noqa: F401

def find_checkpoint(variant: str = "mix") -> str:
    candidates = [
        str(cfg.WORK_ROOT / "output" / f"{cfg.MODEL_NAME}_base_{variant}" / "weights" / "best.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No checkpoint found. Tried: {candidates}\n"
        f"Run 'bash baseline.sh train {variant}' first."
    )


def main(checkpoint=None, *, prepare_data=True, **overrides) -> None:
    cfg.apply_overrides(**overrides)
    cfg.print_config()
    if not prepare_data:
        raise NotImplementedError("Set prepare_data=True")

    df = build_dataset(base_dir=cfg.BASE_DIR, work_dir=cfg.WORK_DIR,
                       data_variant=cfg.DATA_VARIANT, num_classes=cfg.NUM_CLASSES,
                       splits=cfg.SPLITS)
    train_df, valid_df, test_df = split_dataframes(df)
    loaders = build_dataloaders(train_df, valid_df, test_df,
                                batch_size=cfg.BATCH_SIZE, work_dir=cfg.WORK_DIR)

    out_dir = cfg.WORK_ROOT / "output" / f"{cfg.MODEL_NAME}_base_{cfg.DATA_VARIANT}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DetectionModel.create(cfg.MODEL_NAME, num_classes=cfg.NUM_CLASSES,
                                  class_names=cfg.CLASS_NAMES, device=cfg.DEVICE)
    model.load(checkpoint or find_checkpoint(cfg.DATA_VARIANT))

    test_metrics = model.evaluate(iou_threshold=cfg.IOU_THRESHOLD,
                                   score_threshold=cfg.SCORE_THRESHOLD,
                                   base_dir=cfg.WORK_DIR, split="test")
    model.print_metrics(test_metrics)
    model.plot_metrics_per_class(test_metrics,
                                 save_path=str(out_dir / "test_metrics_per_class.png"))
    model.plot_training_curves(save_path=str(out_dir / "training_curves.png"))

    dets, gts = collect_predictions(model, loaders["test_dataset"], score_threshold=0.01)
    plot_confusion_matrix(model, dets, gts, score_thr=cfg.SCORE_THRESHOLD,
                          iou_thr=cfg.IOU_THRESHOLD,
                          save_path=str(out_dir / "test_confusion_matrix.png"))
    plot_pr_curves(model, dets, gts, test_metrics, iou_thr=cfg.IOU_THRESHOLD,
                   save_path=str(out_dir / "test_pr_curves.png"))
    plot_f1_curves(model, dets, gts, iou_thr=cfg.IOU_THRESHOLD,
                   save_path=str(out_dir / "test_f1_curves.png"))
    export_predictions_csv(cfg.MODEL_NAME, dets, gts,
                           image_ids=loaders["test_dataset"].image_ids,
                           iou_thr=cfg.IOU_THRESHOLD, score_thr=cfg.SCORE_THRESHOLD,
                           csv_out=str(out_dir / "test_predictions.csv"),
                           label_to_name=cfg.LABEL_TO_NAME)
    model.visualize_predictions(test_df, cfg.WORK_DIR, n_examples=cfg.N_EXAMPLES,
                                save_dir=str(out_dir / "test_predictions"),
                                score_threshold=cfg.SCORE_THRESHOLD)


if __name__ == "__main__":
    main()
