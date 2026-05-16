"""Ultralytics-backed detector with custom-YAML (SimAM) support."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml as _yaml

from . import config
from .config import BASE_DIR
from core.detection_base import DetectionModel


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
ULTRALYTICS_MODELS: dict = {}
for size in ["n", "s", "m", "l", "x"]:
    ULTRALYTICS_MODELS[f"yolov5{size}"]  = {"weights": f"yolov5{size}u.pt"}
    ULTRALYTICS_MODELS[f"yolov8{size}"]  = {"weights": f"yolov8{size}.pt"}
    ULTRALYTICS_MODELS[f"yolov11{size}"] = {"weights": f"yolo11{size}.pt"}
    ULTRALYTICS_MODELS[f"yolov12{size}"] = {"weights": f"yolo12{size}.pt"}
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
        custom_yaml = self.config.get("custom_yaml")
        if custom_yaml:
            self.model = YOLO(custom_yaml)
            pretrained = self.config.get("weights")
            if pretrained and self._is_valid_pt(pretrained):
                self.model.load(pretrained)
                print(f"[Custom] Loaded {self.model_name} from {custom_yaml} + {pretrained}")
            else:
                print(f"[Custom] Loaded {self.model_name} from {custom_yaml} (no pretrained)")
        else:
            weights = self._resolve_weights(self.config["weights"])
            self.model = YOLO(weights)
            print(f"[UltralyticsDetector] Loaded {self.model_name} ({weights})")

    def _create_data_yaml(self, base_dir, subfolder: str | None = None) -> str:
        # Auto-pick val vs valid based on what exists on disk.
        def _pick(root: Path, candidates: list[str]) -> str:
            for c in candidates:
                if (root / c / "images").is_dir():
                    return f"{c}/images"
            return f"{candidates[0]}/images"

        root = Path(base_dir)
        train_split = _pick(root, ["train"])
        val_split = _pick(root, ["val", "valid"])
        test_split = _pick(root, ["test", "val", "valid"])

        data_dict = {
            "path": str(root),
            "train": train_split,
            "val": val_split,
            "test": test_split,
            "nc": self.num_classes,
            "names": list(self.class_names),
        }
        out_dir = config.WORK_ROOT / "output" / (subfolder or self.model_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = str(out_dir / "yolo_data.yaml")
        with open(yaml_path, "w") as f:
            _yaml.dump(data_dict, f)
        print(f"[data.yaml] train={train_split}  val={val_split}  test={test_split}")
        return yaml_path

    # -- training ---------------------------------------------------------
    def train(self, epochs: int = 75, lr: float = 0.001,
              base_dir=None, imgsz: int = 640, batch: int = 24, patience: int = 15,
              optimizer: str = "SGD", workers: int = 2,
              augmentation: dict | None = None, output_name: str | None = None, **kwargs) -> None:
        from ultralytics import YOLO, settings as ul_settings
        ul_settings.update({
            "mlflow": False, "neptune": False, "comet": False,
            "tensorboard": False, "clearml": False, "wandb": False,
        })

        base_dir = base_dir or config.WORK_DIR
        aug = augmentation or {}
        kwargs = {**aug, **kwargs}

        run_name = output_name or self.model_name
        self.yolo_data_yaml = self._create_data_yaml(base_dir, subfolder=run_name)
        device_str, _workers = _get_device_str(workers)
        self.yolo_run_dir = str(config.WORK_ROOT / "output" / run_name)

        # Phase resume: copy best.pt out of save_dir before reloading
        # (ultralytics may clear save_dir/weights/ on exist_ok=True restart).
        best_pt = Path(self.yolo_run_dir) / "weights" / "best.pt"
        if best_pt.exists():
            safe_resume = config.WORK_ROOT / "output" / f"{run_name}_resume.pt"
            shutil.copy(str(best_pt), str(safe_resume))
            self.model = YOLO(str(safe_resume))
            print(f"[Phase resume] Copied {best_pt} -> {safe_resume} (DDP-safe)")

        self.model.train(
            data=self.yolo_data_yaml, epochs=epochs, lr0=lr,
            imgsz=imgsz, batch=batch, patience=patience, optimizer=optimizer,
            device=device_str, workers=_workers,
            project=str(config.WORK_ROOT / "output"), name=run_name,
            exist_ok=True, verbose=True, **kwargs,
        )
        print(f"\n{self.model_name} phase finished (epochs={epochs}, lr={lr})")

    # -- evaluation -------------------------------------------------------
    def evaluate(self, iou_threshold: float = 0.5, score_threshold: float = 0.3,
                 base_dir=None, imgsz: int = 640, batch: int = 16,
                 split: str = "test") -> dict:
        base_dir = base_dir or config.WORK_DIR
        val_name = os.path.basename(self.yolo_run_dir) if self.yolo_run_dir else self.model_name
        if not self.yolo_data_yaml:
            self.yolo_data_yaml = self._create_data_yaml(base_dir, subfolder=val_name)
        device_str, _ = _get_device_str()

        yolo_metrics = self.model.val(
            data=self.yolo_data_yaml, split=split, imgsz=imgsz, batch=batch,
            device=device_str, plots=True, verbose=True,
            conf=score_threshold, iou=iou_threshold, augment=True,
            project=str(config.WORK_ROOT / "output"), name=val_name, exist_ok=True,
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

    def predict(self, images, img_size: int = 640, conf: float = 0.25, **kwargs):
        """Predict on images. Returns a list of per-image detection dicts.

        Each per-image entry is a list of ``{"bbox", "confidence", "class_id",
        "class_name"}`` dicts (matches the V4 notebook's downstream consumers).
        """
        device_str, _ = _get_device_str()

        # Accept tensors (CHW float in [0,1]) by converting to HWC uint8 first
        if torch.is_tensor(images):
            img_np = images.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
            sources = img_np
        else:
            sources = images

        results = self.model.predict(source=sources, imgsz=img_size, conf=conf,
                                     device=device_str, verbose=False, **kwargs)
        all_preds = []
        for r in results:
            boxes = r.boxes
            preds = []
            if boxes is not None and len(boxes):
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    cid = int(boxes.cls[i].cpu())
                    preds.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(boxes.conf[i].cpu()),
                        "class_id": cid,
                        "class_name": self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid),
                    })
            all_preds.append(preds)
        return all_preds

    # -- IO ---------------------------------------------------------------
    def save(self, path: str) -> None:
        if self.model is not None:
            best = (Path(self.yolo_run_dir) / "weights" / "best.pt"
                    if self.yolo_run_dir else None)
            if best and best.exists():
                shutil.copy(best, path)
                print(f"Saved best weights to {path}")
            else:
                self.model.save(path)
                print(f"Saved to {path}")

    def load(self, path: str) -> None:
        from ultralytics import YOLO
        self.model = YOLO(path)
        norm = os.path.normpath(path)
        if os.path.basename(os.path.dirname(norm)) == "weights":
            self.yolo_run_dir = os.path.dirname(os.path.dirname(norm))
            print(f"Loaded from {path} (run_dir={self.yolo_run_dir})")
        else:
            print(f"Loaded from {path}")

    def plot_training_curves(self, save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        results_csv = (Path(self.yolo_run_dir) / "results.csv"
                       if self.yolo_run_dir else None)
        if results_csv and results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
                if col in df.columns:
                    axes[0].plot(df[col], label=col.split("/")[1])
            axes[0].set_title("Training Losses"); axes[0].legend(); axes[0].grid(True)
            for col in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]:
                if col in df.columns:
                    axes[1].plot(df[col], label=col.split("/")[1])
            axes[1].set_title("Validation Losses"); axes[1].legend(); axes[1].grid(True)
            for col in ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
                if col in df.columns:
                    axes[2].plot(df[col], label=col.split("/")[1])
            axes[2].set_title("mAP Metrics"); axes[2].legend(); axes[2].grid(True)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.show()
        else:
            print("No results.csv found.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def register_all() -> None:
    for name, cfg in ULTRALYTICS_MODELS.items():
        DetectionModel.register(name, UltralyticsDetector, cfg)
    print(f"Registered {len(ULTRALYTICS_MODELS)} ultralytics models")


def register_simam(model_name: str, custom_yaml: str, pretrained: str | None = None) -> str:
    """Register a SimAM variant of ``model_name`` (e.g. yolov12m-simam)."""
    custom_name = f"{model_name}-simam"
    base_w = pretrained if pretrained and Path(pretrained).is_file() else ULTRALYTICS_MODELS[model_name]["weights"]
    ULTRALYTICS_MODELS[custom_name] = {"weights": base_w, "custom_yaml": custom_yaml}
    DetectionModel.register(custom_name, UltralyticsDetector, ULTRALYTICS_MODELS[custom_name])
    print(f"Registered custom model: {custom_name}")
    print(f"  yaml    = {custom_yaml}")
    print(f"  weights = {base_w}")
    return custom_name


# Auto-register stock models on import.
register_all()
