"""Confusion matrix, PR-curve, F1-curve plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .metrics import (
    build_confusion_matrix, compute_ap, f1_from_ranking,
    pr_curve_ranking, smooth_pr_ultralytics,
)


_PALETTE = [
    "#5B9BD5", "#ED7D31", "#70AD47", "#C0504D", "#9F8FD9",
    "#FFC000", "#4472C4", "#264478", "#43682B", "#7F3B17",
]


def plot_confusion_matrix(mdl, dets, gts,
                          score_thr: float = 0.25, iou_thr: float = 0.5,
                          save_path: str | None = None) -> None:
    names = list(mdl.class_names) + ["BG"]
    cm = build_confusion_matrix(dets, gts, mdl.num_classes, iou_thr, score_thr)
    cm_n = np.divide(
        cm.astype(float), cm.sum(1, keepdims=True),
        where=cm.sum(1, keepdims=True) > 0,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = LinearSegmentedColormap.from_list("wb", ["#ffffff", "#1565C0"])
    im = ax.imshow(cm_n, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    n = mdl.num_classes + 1
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Ground Truth", fontweight="bold")
    ax.set_title(f"{mdl.model_name} — Confusion Matrix (Normalized)", fontweight="bold")

    thr = cm_n.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_n[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_n[i, j] > thr else "black")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pr_curves(mdl, dets, gts, metrics: dict,
                   iou_thr: float = 0.5, save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    all_p_smooth = []
    r_grid = np.linspace(0, 1, 1000)

    for i in range(mdl.num_classes):
        prec, rec, _ = pr_curve_ranking(dets, gts, i + 1, iou_thr)
        ap = compute_ap(prec, rec)
        p_smooth, _ = smooth_pr_ultralytics(prec, rec, num_points=1000)
        all_p_smooth.append(p_smooth)
        c = _PALETTE[i % len(_PALETTE)]
        ax.plot(r_grid, p_smooth, color=c, lw=2,
                label=f"{mdl.class_names[i]} {ap:.3f}")

    mean_p = np.mean(all_p_smooth, axis=0)
    mAP = metrics.get("mAP", 0.0)
    ax.plot(r_grid, mean_p, color="#000080", lw=3,
            label=f"all classes {mAP:.3f} mAP@{iou_thr}")

    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_f1_curves(mdl, dets, gts,
                   iou_thr: float = 0.5, save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    all_f1 = []
    conf_grid = np.linspace(0, 1, 1000)

    for i in range(mdl.num_classes):
        f1, _ = f1_from_ranking(dets, gts, i + 1, iou_thr, num_points=1000)
        all_f1.append(f1)
        c = _PALETTE[i % len(_PALETTE)]
        ax.plot(conf_grid, f1, color=c, lw=2, label=f"{mdl.class_names[i]}")

    mean_f1 = np.mean(all_f1, axis=0)
    bi = np.argmax(mean_f1)
    ax.plot(conf_grid, mean_f1, color="#000080", lw=3,
            label=f"all classes {mean_f1[bi]:.2f} at {conf_grid[bi]:.3f}")

    ax.set(xlabel="Confidence", ylabel="F1", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("F1-Confidence Curve", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
