"""Bounding-box rendering and EDA plots."""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from . import config
from .config import CLASS_NAMES, LABEL_TO_NAME, NUM_CLASSES


COLOR_MAP = plt.cm.get_cmap("tab10", NUM_CLASSES)


def draw_bboxes(ax, rows_or_boxes, labels=None, scores=None) -> None:
    if isinstance(rows_or_boxes, pd.DataFrame):
        for _, row in rows_or_boxes.iterrows():
            cls_id = int(row["class"])
            if cls_id <= 0:
                continue
            x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            color = COLOR_MAP(cls_id - 1)
            ax.add_patch(matplotlib.patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, lw=2, ec=color, fc="none"))
            ax.text(x1, y1 - 3, LABEL_TO_NAME.get(cls_id, "?"),
                    color=color, fontsize=9, weight="bold",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5))
        return

    for i, box in enumerate(np.array(rows_or_boxes)):
        x1, y1, x2, y2 = box
        cls_id = int(labels[i]) if labels is not None else 0
        color = COLOR_MAP(cls_id - 1) if 0 < cls_id <= NUM_CLASSES else "lime"
        ax.add_patch(matplotlib.patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, lw=2, ec=color, fc="none"))
        name = LABEL_TO_NAME.get(cls_id, str(cls_id))
        if scores is not None:
            name = f"{name} {float(scores[i]):.2f}"
        ax.text(x1, y1 - 3, name, color=color, fontsize=9, weight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5))


def show_image(img, rows_or_boxes=None, labels=None, scores=None, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(18, 10))
    if torch.is_tensor(img):
        img = img.permute(1, 2, 0).cpu().numpy()
    ax.imshow(img, cmap="binary")
    if rows_or_boxes is not None:
        draw_bboxes(ax, rows_or_boxes, labels, scores)
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def plot_class_distribution(df: pd.DataFrame, save_path: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    class_counts = df["class"].map(LABEL_TO_NAME).value_counts().sort_index()
    colors_eda = sns.color_palette("Set2", len(CLASS_NAMES))
    class_counts.plot(kind="bar", ax=axes[0], color=colors_eda, edgecolor="black")
    axes[0].set_title("Overall Defect Class Distribution", fontsize=14, weight="bold")
    axes[0].set_xlabel("Defect Class")
    axes[0].set_ylabel("Number of Annotations")
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=10, weight="bold")

    pivot_eda = df.copy()
    pivot_eda["class_name"] = pivot_eda["class"].map(LABEL_TO_NAME)
    ct = pivot_eda.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    ct = ct.reindex(columns=sorted(ct.columns))
    ct.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2", edgecolor="black")
    axes[1].set_title("Class Distribution per Split", fontsize=14, weight="bold")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Number of Annotations")
    axes[1].legend(title="Defect", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sample_grid(train_df: pd.DataFrame, work_dir=None, n: int = 9, seed: int = 42) -> None:
    work_dir = work_dir or config.WORK_DIR
    rng = np.random.default_rng(seed)
    files = train_df["file"].unique()
    chosen = rng.choice(files, size=min(n, len(files)), replace=False)

    rows = cols = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx >= len(chosen):
            ax.axis("off")
            continue
        fname = chosen[idx]
        img = plt.imread(os.path.join(work_dir, "train", "images", fname + ".jpg"))
        show_image(img, train_df[train_df["file"] == fname], ax=ax, title=fname)
    plt.suptitle("Sample Training Images with Ground Truth", fontsize=16, weight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
