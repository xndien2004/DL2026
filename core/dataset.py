"""PyTorch Dataset + DataLoader wrappers around the prepared YOLO directory."""

from __future__ import annotations

import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


class DrillData(torch.utils.data.Dataset):
    """One entry per unique image; bboxes aggregated from dataframe rows."""

    def __init__(self, df: pd.DataFrame, img_dir: str, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_ids = df["file"].unique().tolist()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        fid = self.image_ids[idx]
        rows = self.df[self.df["file"] == fid]
        img_path = os.path.join(self.img_dir, fid + ".jpg")
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        boxes = rows[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
        labels = rows["class"].values
        target = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        if self.transforms:
            s = self.transforms(image=img, bboxes=boxes, labels=labels.tolist())
            img = s["image"]
            target["boxes"] = torch.stack(tuple(map(torch.tensor, zip(*s["bboxes"])))).permute(1, 0)
            target["labels"] = torch.tensor(s["labels"], dtype=torch.int64)
        return torch.tensor(img), target, fid


def get_transform() -> A.Compose:
    return A.Compose(
        [ToTensorV2(p=1.0)],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def collate_fn(batch):
    return tuple(zip(*batch))


def make_dataset(df: pd.DataFrame, split: str, work_dir) -> DrillData:
    return DrillData(df, os.path.join(work_dir, split, "images") + "/", get_transform())


def build_dataloaders(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame,
                      batch_size: int, num_workers: int = 6, work_dir=None) -> dict:
    train_dataset = make_dataset(train_df, "train", work_dir=work_dir)
    valid_dataset = make_dataset(valid_df, "val", work_dir=work_dir)
    test_dataset = make_dataset(test_df, "test", work_dir=work_dir)
    if len(valid_dataset) == 0:
        raise ValueError("valid_dataset is empty. Check split names and re-run data preparation.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)
    print(f"Loaders: train={len(train_loader)}, val={len(valid_loader)}, test={len(test_loader)}")
    return {
        "train_dataset": train_dataset, "valid_dataset": valid_dataset, "test_dataset": test_dataset,
        "train_loader": train_loader, "valid_loader": valid_loader, "test_loader": test_loader,
    }
