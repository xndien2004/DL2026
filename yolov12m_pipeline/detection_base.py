"""Abstract DetectionModel base class with a model registry."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from tqdm.auto import tqdm

from .config import CLASS_NAMES, DEVICE
from .visualization import show_image


class DetectionModel(ABC):
    """Common interface (train/predict/evaluate/visualize) for all detector backends."""

    _registry: dict[str, tuple[type, dict]] = {}

    def __init__(self, model_name: str, num_classes: int, class_names, device, **kwargs):
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.model = None
        self.train_losses: list[float] = []
        self.val_maps: list[float] = []
        self.best_map = 0.0

    # -- registry ----------------------------------------------------------
    @classmethod
    def register(cls, name: str, subclass: type, config: dict) -> None:
        cls._registry[name] = (subclass, config)

    @classmethod
    def create(cls, model_name: str, num_classes: int = 5,
               class_names=None, device=None, **kwargs) -> "DetectionModel":
        class_names = class_names or CLASS_NAMES
        device = device or DEVICE
        if model_name not in cls._registry:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {sorted(cls._registry.keys())}"
            )
        subclass, config = cls._registry[model_name]
        return subclass(model_name, num_classes, class_names, device, config=config, **kwargs)

    # -- to be implemented by subclasses ----------------------------------
    @abstractmethod
    def _build_model(self): ...

    @abstractmethod
    def train(self, epochs: int, lr: float, **kwargs): ...

    @abstractmethod
    def predict(self, image_tensor): ...

    # -- shared helpers ---------------------------------------------------
    @staticmethod
    def _to_numpy(v):
        return v.cpu().numpy() if torch.is_tensor(v) else np.array(v)

    def evaluate(self, dataset, iou_threshold: float = 0.5, score_threshold: float = 0.3):
        """Per-class precision / recall / F1 / AP@iou + mean inference time."""
        self.model.eval()
        detections, gts, inference_times = [], [], []

        for i in tqdm(range(len(dataset)), desc=f"Evaluating {self.model_name}"):
            img, target, _ = dataset[i]
            start = time.time()
            pred = self.predict(img)
            inference_times.append(time.time() - start)

            keep = pred["scores"] >= score_threshold
            detections.append({k: self._to_numpy(pred[k][keep]) for k in ("boxes", "labels", "scores")})
            gts.append({k: self._to_numpy(target[k]) for k in ("boxes", "labels")})

        results, aps = {}, []
        for cls in range(1, self.num_classes + 1):
            preds_cls, total_gt = [], 0
            for img_idx in range(len(dataset)):
                total_gt += (gts[img_idx]["labels"] == cls).sum()
                for j in np.where(detections[img_idx]["labels"] == cls)[0]:
                    preds_cls.append({
                        "score": detections[img_idx]["scores"][j],
                        "box": detections[img_idx]["boxes"][j],
                        "img": img_idx,
                    })
            preds_cls.sort(key=lambda x: x["score"], reverse=True)

            tp, fp = np.zeros(len(preds_cls)), np.zeros(len(preds_cls))
            matched = {i: set() for i in range(len(dataset))}
            for i, p in enumerate(preds_cls):
                gt = gts[p["img"]]
                mask = gt["labels"] == cls
                gt_boxes, gt_ids = gt["boxes"][mask], np.where(mask)[0]
                if len(gt_boxes) == 0:
                    fp[i] = 1
                    continue
                ious = torchvision.ops.box_iou(
                    torch.tensor(p["box"]).unsqueeze(0),
                    torch.tensor(gt_boxes),
                ).numpy()[0]
                best = np.argmax(ious)
                if ious[best] >= iou_threshold and gt_ids[best] not in matched[p["img"]]:
                    tp[i] = 1
                    matched[p["img"]].add(gt_ids[best])
                else:
                    fp[i] = 1

            tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
            recall = tp_cum / max(total_gt, 1)
            precision = tp_cum / (tp_cum + fp_cum + 1e-6)
            ap = sum(np.max(precision[recall >= t]) if np.any(recall >= t) else 0
                     for t in np.arange(0.0, 1.1, 0.1)) / 11.0
            aps.append(ap)
            t_tp, t_fp = int(tp.sum()), int(fp.sum())
            p = t_tp / max(t_tp + t_fp, 1)
            r = t_tp / max(total_gt, 1)
            results[cls] = {
                "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-6),
                "ap": ap, "gt": int(total_gt), "tp": t_tp, "fp": t_fp,
            }

        results["mAP"] = float(np.mean(aps))
        results["avg_inference_time"] = float(np.mean(inference_times))
        return results

    def print_metrics(self, metrics: dict) -> None:
        bar = "=" * 70
        print(f"\n{bar}\nEVALUATION: {self.model_name}\n{bar}")
        print(f"{'Class':<20} {'P':>8} {'R':>8} {'F1':>8} {'AP@0.5':>10} {'GT':>6} {'TP':>6} {'FP':>6}")
        print("-" * 70)
        for cls in range(1, self.num_classes + 1):
            m = metrics[cls]
            name = self.class_names[cls - 1] if cls - 1 < len(self.class_names) else str(cls)
            print(f"{name:<20} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
                  f"{m['ap']:>10.4f} {m['gt']:>6d} {m['tp']:>6d} {m['fp']:>6d}")
        print("-" * 70)
        vals = lambda k: [metrics[c][k] for c in range(1, self.num_classes + 1)]
        print(f"{'Mean':<20} {np.mean(vals('precision')):>8.4f} {np.mean(vals('recall')):>8.4f} "
              f"{np.mean(vals('f1')):>8.4f} {metrics['mAP']:>10.4f}")
        avg_t = metrics["avg_inference_time"]
        if avg_t > 0:
            print(f"\nmAP@0.5 = {metrics['mAP']:.4f}  |  {avg_t * 1000:.2f} ms/img ({1 / avg_t:.2f} FPS)")

    def plot_training_curves(self, save_path: str = "training_curves.png") -> None:
        if not self.train_losses:
            print("No training data to plot.")
            return
        epochs = range(1, len(self.train_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(epochs, self.train_losses, marker="o")
        ax1.set(title=f"{self.model_name} — Train Loss", xlabel="Epoch", ylabel="Loss")
        ax1.grid(True)
        if self.val_maps:
            ax2.plot(epochs, self.val_maps, marker="o", color="green")
        ax2.set(title=f"{self.model_name} — Val mAP@0.5", xlabel="Epoch", ylabel="mAP", ylim=(0, 1))
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()

    def visualize_predictions(self, test_df, base_dir, n_examples: int = 10,
                              save_dir: str = "predictions", score_threshold: float = 0.3) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        for sample_file in test_df["file"].unique()[:n_examples]:
            img_path = os.path.join(base_dir, "test", "images", sample_file + ".jpg")
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            tensor = T.ToTensor()((img / 255.0).astype(np.float32)).float()
            pred = self.predict(tensor)
            keep = pred["scores"] >= score_threshold
            boxes, labels, scores = [self._to_numpy(pred[k][keep]) for k in ("boxes", "labels", "scores")]

            fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 6))
            show_image(img, test_df[test_df["file"] == sample_file], ax=ax_gt, title="Ground Truth")
            show_image(img, boxes, labels, scores, ax=ax_pred, title=f"Prediction ({self.model_name})")
            short = sample_file.split(".rf.")[0] if ".rf." in sample_file else sample_file
            fig.suptitle(short[-50:], fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{sample_file}.png"), bbox_inches="tight", dpi=120)
            plt.show()
        print(f"Saved {min(n_examples, test_df['file'].nunique())} prediction images to {save_dir}")

    def plot_metrics_per_class(self, metrics: dict,
                               save_path: str = "test_metrics_per_class.png") -> None:
        class_ids = list(range(1, self.num_classes + 1))
        names = [self.class_names[i] for i in range(self.num_classes)]
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_ids)))
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, key, title in zip(axes,
                                  ["precision", "recall", "f1", "ap"],
                                  ["Precision", "Recall", "F1-score", "AP@0.5"]):
            vals = [metrics[c][key] for c in class_ids]
            bars = ax.bar(range(len(class_ids)), vals, color=colors)
            ax.set_xticks(range(len(class_ids)))
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.axhline(y=np.mean(vals), linestyle="--", alpha=0.7,
                       label=f"Mean: {np.mean(vals):.3f}")
            ax.legend()
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", fontsize=9)
        plt.suptitle(f"{self.model_name} — Test Metrics per Class", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def save(self, path: str) -> None:
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            print(f"Saved to {path}")

    def load(self, path: str) -> None:
        self._build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device).eval()
        print(f"Loaded from {path}")
