"""Evaluation entry — V4 SimAM pipeline."""
from __future__ import annotations
import os
from . import config as cfg
from core.data_preparation import build_dataset, split_dataframes
from core.dataset import build_dataloaders
from core.detection_base import DetectionModel
from core.evaluation import export_predictions_csv
from core.metrics import collect_predictions
from core.plots import plot_confusion_matrix, plot_f1_curves, plot_pr_curves
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


def find_checkpoint(model_name: str) -> str:
    variant = cfg.DATA_VARIANT
    candidates = [
        str(cfg.WORK_ROOT / "output" / f"yolov12m_new_{variant}" / "weights" / "best.pt"),
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

    df = build_dataset(base_dir=cfg.BASE_DIR, work_dir=cfg.WORK_DIR,
                       data_variant=cfg.DATA_VARIANT, num_classes=cfg.NUM_CLASSES,
                       splits=cfg.SPLITS)
    train_df, valid_df, test_df = split_dataframes(df)

    loaders = build_dataloaders(train_df, valid_df, test_df,
                                batch_size=cfg.BATCH_SIZE, work_dir=cfg.WORK_DIR)

    train_model_name = _setup_model_name()
    out_dir = cfg.WORK_ROOT / "output" / f"yolov12m_new_{cfg.DATA_VARIANT}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DetectionModel.create(train_model_name, num_classes=cfg.NUM_CLASSES,
                                  class_names=cfg.CLASS_NAMES, device=cfg.DEVICE)
    model.load(checkpoint or find_checkpoint(train_model_name))
    # Always re-create the data.yaml so it points at the local WORK_DIR.
    model.yolo_data_yaml = model._create_data_yaml(cfg.WORK_DIR)

    test_metrics = model.evaluate(iou_threshold=cfg.IOU_THRESHOLD,
                                   score_threshold=cfg.SCORE_THRESHOLD,
                                   base_dir=cfg.WORK_DIR, split="test")
    model.print_metrics(test_metrics)
    model.plot_metrics_per_class(test_metrics,
                                 save_path=str(out_dir / "test_metrics_per_class.png"))
    model.plot_training_curves(save_path=str(out_dir / "training_curves.png"))

    print("Collecting raw predictions...")
    dets, gts = collect_predictions(model, loaders["test_dataset"], score_threshold=0.01)
    print(f"Done — {len(dets)} images")

    plot_confusion_matrix(model, dets, gts,
                          score_thr=cfg.SCORE_THRESHOLD, iou_thr=cfg.IOU_THRESHOLD,
                          save_path=str(out_dir / "test_confusion_matrix.png"))
    plot_pr_curves(model, dets, gts, test_metrics,
                   iou_thr=cfg.IOU_THRESHOLD,
                   save_path=str(out_dir / "test_pr_curves.png"))
    plot_f1_curves(model, dets, gts,
                   iou_thr=cfg.IOU_THRESHOLD,
                   save_path=str(out_dir / "test_f1_curves.png"))

    export_predictions_csv(train_model_name, dets, gts,
                           image_ids=loaders["test_dataset"].image_ids,
                           iou_thr=cfg.IOU_THRESHOLD, score_thr=cfg.SCORE_THRESHOLD,
                           csv_out=str(out_dir / "test_predictions.csv"),
                           label_to_name=cfg.LABEL_TO_NAME)

    model.visualize_predictions(test_df, cfg.WORK_DIR, n_examples=cfg.N_EXAMPLES,
                                save_dir=str(out_dir / "test_predictions"),
                                score_threshold=cfg.SCORE_THRESHOLD)


if __name__ == "__main__":
    main()
