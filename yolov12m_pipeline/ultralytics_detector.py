"""Ultralytics-backed detector (YOLOv5/8/11/12/26 + RT-DETR) and registry registration."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import torch
import yaml

from .config import BASE_DIR
from core.detection_base import DetectionModel


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
ULTRALYTICS_MODELS: dict[str, dict] = {}
for size in ["n", "s", "m", "l", "x"]:
    ULTRALYTICS_MODELS[f"yolov5{size}"] = {"weights": f"yolov5{size}u.pt"}
    ULTRALYTICS_MODELS[f"yolov8{size}"] = {"weights": f"yolov8{size}.pt"}
    ULTRALYTICS_MODELS[f"yolov11{size}"] = {"weights": f"yolo11{size}.pt"}
    ULTRALYTICS_MODELS[f"yolov12{size}"] = {"weights": f"yolo12{size}.pt"}
    ULTRALYTICS_MODELS[f"yolov26{size}"] = {"weights": f"yolo26{size}.pt"}
ULTRALYTICS_MODELS["rtdetr-l"] = {"weights": "rtdetr-l.pt"}
ULTRALYTICS_MODELS["rtdetr-x"] = {"weights": "rtdetr-x.pt"}


def _get_device_str(workers: int = 2) -> tuple[str | int, int]:
    n = torch.cuda.device_count()
    if n >= 2:
        return "0,1", min(workers * n, 8)
    if n == 1:
        return 0, workers
    return "cpu", workers


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class UltralyticsDetector(DetectionModel):
    """Wraps Ultralytics' YOLO/RT-DETR API in our DetectionModel interface."""

    def __init__(self, model_name, num_classes, class_names, device, config=None, **kwargs):
        super().__init__(model_name, num_classes, class_names, device, **kwargs)
        self.config = config or ULTRALYTICS_MODELS[model_name]
        self.yolo_data_yaml = None
        self.yolo_run_dir = None
        self._build_model()

    @staticmethod
    def _is_valid_pt(path: str) -> bool:
        import zipfile
        try:
            with zipfile.ZipFile(path, "r"):
                return True
        except Exception:
            return False

    def _resolve_weights(self, weights: str) -> str:
        from ultralytics.utils.downloads import attempt_download_asset
        if os.path.exists(weights):
            if self._is_valid_pt(weights):
                return weights
            print(f"[UltralyticsDetector] '{weights}' exists but is corrupt "
                  f"(possibly a git-lfs pointer). Deleting and re-downloading...")
            os.remove(weights)
        try:
            local = attempt_download_asset(weights)
            if local and os.path.exists(local):
                return local
        except Exception as e:
            print(f"[UltralyticsDetector] WARNING download failed for {weights}: {e}")
        raise FileNotFoundError(
            f"Weights '{weights}' not found or corrupt. "
            f"Download failed — retry or place the .pt file in the working directory."
        )

    def _build_model(self) -> None:
        from ultralytics import YOLO
        weights = self._resolve_weights(self.config["weights"])
        self.model = YOLO(weights)
        print(f"[UltralyticsDetector] Loaded {self.model_name} ({weights})")

    def _create_data_yaml(self, base_dir) -> str:
        cfg = {
            "path": str(base_dir),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(self.class_names),
            "names": list(self.class_names),
        }
        yaml_path = f"{self.model_name}_data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        return yaml_path

    # -- training ---------------------------------------------------------
    def train(self, epochs: int = 20, lr: float = 0.001,
              base_dir=None, imgsz: int = 640, batch: int = 16, patience: int = 10,
              optimizer: str = "AdamW", weight_decay: float = 0.0005, workers: int = 2,
              augmentation: dict | None = None, **kwargs) -> None:
        from ultralytics import YOLO
        base_dir = base_dir or BASE_DIR
        self.yolo_data_yaml = self._create_data_yaml(base_dir)
        device_str, _workers = _get_device_str(workers)
        aug = augmentation or {}

        train_kwargs = dict(
            data=self.yolo_data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
            lr0=lr, lrf=0.01, optimizer=optimizer, weight_decay=weight_decay,
            patience=patience, save=True, save_period=max(epochs // 4, 1),
            project="runs", name=self.model_name, exist_ok=True,
            device=device_str, workers=_workers, seed=42, verbose=True, plots=True,
            hsv_h=aug.get("hsv_h", 0.015), hsv_s=aug.get("hsv_s", 0.7),
            hsv_v=aug.get("hsv_v", 0.4), degrees=aug.get("degrees", 10.0),
            translate=aug.get("translate", 0.1), scale=aug.get("scale", 0.5),
            fliplr=aug.get("fliplr", 0.5), flipud=aug.get("flipud", 0.0),
            mosaic=aug.get("mosaic", 1.0), mixup=aug.get("mixup", 0.1),
            copy_paste=aug.get("copy_paste", 0.0), erasing=aug.get("erasing", 0.0),
        )
        train_kwargs.update(kwargs)
        self.model.train(**train_kwargs)

        self.yolo_run_dir = str(self.model.trainer.save_dir)
        best_path = os.path.join(self.yolo_run_dir, "weights", "best.pt")
        if os.path.exists(best_path):
            self.model = YOLO(best_path)
            print(f"  Loaded best weights from {best_path}")
        self._load_training_csv()
        print(f"\n{self.model_name} training completed!")

    def _load_training_csv(self) -> None:
        candidates = [self.yolo_run_dir, os.path.join("runs", self.model_name)]
        for d in filter(None, candidates):
            csv_path = os.path.join(d, "results.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            if "train/box_loss" in df.columns:
                self.train_losses = df["train/box_loss"].tolist()
            if "metrics/mAP50(B)" in df.columns:
                self.val_maps = df["metrics/mAP50(B)"].tolist()
            if self.train_losses:
                self.yolo_run_dir = d
                break

    # -- evaluation -------------------------------------------------------
    def evaluate(self, iou_threshold: float = 0.5, score_threshold: float = 0.3,
                 base_dir=None, imgsz: int = 640, batch: int = 16, split: str = "val") -> dict:
        base_dir = base_dir or BASE_DIR
        if not self.yolo_data_yaml:
            self.yolo_data_yaml = self._create_data_yaml(base_dir)
        device_str, _ = _get_device_str()

        yolo_metrics = self.model.val(
            data=self.yolo_data_yaml, split=split, imgsz=imgsz, batch=batch,
            device=device_str, plots=True, verbose=True,
            conf=score_threshold, iou=iou_threshold,
        )

        nt = np.array(getattr(yolo_metrics.box, "nt_per_class", None)
                      or getattr(yolo_metrics, "nt_per_class", np.zeros(self.num_classes)))
        results: dict = {}
        for i in range(self.num_classes):
            p = float(yolo_metrics.box.p[i]) if i < len(yolo_metrics.box.p) else 0.0
            r = float(yolo_metrics.box.r[i]) if i < len(yolo_metrics.box.r) else 0.0
            ap50 = float(yolo_metrics.box.ap50[i]) if i < len(yolo_metrics.box.ap50) else 0.0
            f1 = 2 * p * r / max(p + r, 1e-6)
            gt = int(nt[i]) if i < len(nt) else 0
            tp = int(round(r * gt))
            fp = int(round(tp * (1 / max(p, 1e-6) - 1))) if tp > 0 else 0
            results[i + 1] = {"precision": p, "recall": r, "f1": f1, "ap": ap50,
                              "gt": gt, "tp": tp, "fp": fp}

        results["mAP"] = float(yolo_metrics.box.map50)
        results["mAP_50_95"] = float(yolo_metrics.box.map)
        infer_ms = (yolo_metrics.speed or {}).get("inference") if hasattr(yolo_metrics, "speed") else None
        results["avg_inference_time"] = float(infer_ms) / 1000.0 if infer_ms else 0.01
        return results

    def predict(self, image_tensor):
        if torch.is_tensor(image_tensor):
            img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        else:
            img_np = image_tensor
        result = self.model.predict(
            img_np, verbose=False,
            device=0 if torch.cuda.is_available() else "cpu",
        )[0]
        if result.boxes is not None and len(result.boxes) > 0:
            return {
                "boxes": result.boxes.xyxy.cpu(),
                "labels": result.boxes.cls.cpu().int() + 1,
                "scores": result.boxes.conf.cpu(),
            }
        return {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros(0, dtype=torch.int64),
            "scores": torch.zeros(0),
        }

    # -- IO ---------------------------------------------------------------
    def save(self, path: str) -> None:
        if self.yolo_run_dir:
            best = os.path.join(self.yolo_run_dir, "weights", "best.pt")
            if os.path.exists(best):
                shutil.copy2(best, path)
                print(f"Saved to {path}")

    def load(self, path: str) -> None:
        from ultralytics import YOLO
        self.model = YOLO(path)
        norm = os.path.normpath(path)
        if os.path.basename(os.path.dirname(norm)) == "weights":
            self.yolo_run_dir = os.path.dirname(os.path.dirname(norm))
        print(f"Loaded from {path}")

    def plot_training_curves(self, save_path: str = "training_curves.png") -> None:
        if not self.train_losses:
            self._load_training_csv()
        super().plot_training_curves(save_path)


# ---------------------------------------------------------------------------
# Register everything in the catalog
# ---------------------------------------------------------------------------
def register_all() -> None:
    for name, cfg in ULTRALYTICS_MODELS.items():
        DetectionModel.register(name, UltralyticsDetector, cfg)
    print(f"Registered {len(ULTRALYTICS_MODELS)} ultralytics models")


# Auto-register on import so DetectionModel.create("yolov12m") works immediately.
register_all()
