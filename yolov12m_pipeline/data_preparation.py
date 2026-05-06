"""Parse COCO annotations and materialize a YOLO-flat directory layout."""

from __future__ import annotations

import json as _json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config
from .config import (
    BASE_DIR, DATA_VARIANT, IMG_EXTENSIONS, NUM_CLASSES,
    SPLITS, VARIANT_ALIASES, WORK_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_variant(v: str | None) -> str:
    raw = str(v or DATA_VARIANT).strip().lower()
    norm = VARIANT_ALIASES.get(raw, raw)
    assert norm in {"mix", "bright", "dark"}, f"Invalid variant: {norm}"
    return norm


def _resolve_split_root(base_dir: Path | str, split: str) -> Path:
    aliases = {
        "train": ["train"],
        "valid": ["valid", "val"],
        "val": ["val", "valid"],
        "test": ["test"],
    }
    for s in aliases.get(split, [split]):
        p = Path(base_dir) / s
        if p.is_dir():
            return p
    return Path(base_dir) / split


def _infer_lighting(fn: str, path: str) -> str:
    h = (str(path).replace("\\", "/") + " " + str(fn)).lower()
    if "/bright_field/" in h or "_bright_" in h:
        return "bright"
    if "/dark_field/" in h or "_dark_" in h:
        return "dark"
    return "other"


# ---------------------------------------------------------------------------
# COCO parsing
# ---------------------------------------------------------------------------
def parse_coco_collect(base_dir: Path | str, split: str,
                       data_variant: str = DATA_VARIANT) -> tuple[list[dict], dict[str, str]]:
    """Walk a split folder, merge image metadata + COCO annotations.

    Returns (records, file_map) where each record is one bbox row (or a
    blank row if the image has no annotations of interest).
    """
    variant = _norm_variant(data_variant)
    root = _resolve_split_root(base_dir, split)

    cands: list[Path] = []
    if variant in {"mix", "bright"}:
        cands.append(root / "Bright_Field")
    if variant in {"mix", "dark"}:
        cands.append(root / "Dark_Field")
    img_roots = [str(p) for p in cands if p.is_dir()] or ([str(root)] if root.is_dir() else [])
    assert img_roots, f"No image root for {split}/{variant}"

    file_map: dict[str, str] = {}
    for r in img_roots:
        for dp, _, fns in os.walk(r):
            for fn in fns:
                if fn.lower().endswith(IMG_EXTENSIONS):
                    file_map.setdefault(fn, os.path.join(dp, fn))

    mapping = {"bright": ["Bright_Field"], "dark": ["Dark_Field"], "mix": ["Bright_Field", "Dark_Field"]}
    ann = root / "_annotations.coco.json"
    if ann.exists():
        ann_paths = [str(ann)]
    else:
        ann_paths = [
            str(root / f / "_annotations.coco.json")
            for f in mapping[variant]
            if (root / f / "_annotations.coco.json").exists()
        ]
        if not ann_paths:
            ann_paths = [str(p) for p in sorted(root.glob("*/_annotations.coco.json"))]
    assert ann_paths, f"No annotations in: {root}"

    keep_ids = set(range(1, NUM_CLASSES + 1))
    img_meta: dict[str, dict[str, int]] = {}
    raw_anns: dict[str, list] = defaultdict(list)
    for ap in ann_paths:
        with open(ap, "r", encoding="utf-8") as f:
            coco = _json.load(f)
        local = {int(im["id"]): str(im["file_name"]) for im in coco.get("images", [])}
        for im in coco.get("images", []):
            fn = str(im["file_name"])
            if fn not in img_meta:
                img_meta[fn] = {"w": int(im.get("width", 0) or 0), "h": int(im.get("height", 0) or 0)}
        for a in coco.get("annotations", []):
            cls = int(a.get("category_id", -1))
            if cls not in keep_ids:
                continue
            fn = local.get(int(a.get("image_id", -1)))
            bbox = a.get("bbox", [])
            if fn and len(bbox) == 4:
                raw_anns[fn].append((cls, *[float(v) for v in bbox]))

    records: list[dict] = []
    for fn in sorted(img_meta):
        src = file_map.get(fn)
        if not src:
            continue
        lit = _infer_lighting(fn, src)
        if variant != "mix" and lit != variant:
            continue
        stem, ext = os.path.splitext(fn)
        w, h = max(1, img_meta[fn]["w"]), max(1, img_meta[fn]["h"])
        seen, valid = set(), []
        for cls, x, y, bw, bh in raw_anns.get(fn, []):
            key = (int(cls), round(x, 4), round(y, 4), round(bw, 4), round(bh, 4))
            if key in seen:
                continue
            seen.add(key)
            x1, y1 = max(0, min(w, int(round(x)))), max(0, min(h, int(round(y))))
            x2, y2 = max(0, min(w, int(round(x + bw)))), max(0, min(h, int(round(y + bh))))
            if x2 > x1 and y2 > y1:
                valid.append((int(cls), x1, y1, x2, y2))
        base = {"file": stem, "src_path": src, "src_ext": ext, "width": w, "height": h,
                "split": split, "lighting": lit}
        if not valid:
            records.append({**base, "xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0, "class": 0})
        else:
            for cls, x1, y1, x2, y2 in valid:
                records.append({**base, "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2, "class": cls})
    return records, file_map


# ---------------------------------------------------------------------------
# YOLO directory layout
# ---------------------------------------------------------------------------
def prepare_yolo_flat_dir(split: str, file_map: dict[str, str],
                          records: Iterable[dict], work_dir: Path = WORK_DIR) -> None:
    """Symlink (or copy) images and write YOLO-format .txt labels for one split."""
    img_out = os.path.join(work_dir, split, "images")
    lbl_out = os.path.join(work_dir, split, "labels")
    for d in (img_out, lbl_out):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_file[r["file"]].append(r)

    created = 0
    for stem, recs in by_file.items():
        src, ext = recs[0]["src_path"], recs[0].get("src_ext", ".jpg")
        if not src or not os.path.exists(src):
            continue
        dst = os.path.join(img_out, stem + ext)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
            for r in recs:
                if r["class"] == 0:
                    continue
                iw, ih = r["width"], r["height"]
                cx = ((r["xmin"] + r["xmax"]) / 2) / iw
                cy = ((r["ymin"] + r["ymax"]) / 2) / ih
                bw = (r["xmax"] - r["xmin"]) / iw
                bh = (r["ymax"] - r["ymin"]) / ih
                f.write(f"{r['class'] - 1} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        created += 1
    print(f"  {split}: {created} images+labels -> {img_out}")


def build_dataset(base_dir: Path = BASE_DIR, work_dir: Path = WORK_DIR,
                  data_variant: str = DATA_VARIANT) -> pd.DataFrame:
    """Run full data prep: parse COCO + materialize YOLO dirs. Returns annotation DataFrame."""
    all_records: list[dict] = []
    all_file_maps: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        records, all_file_maps[split] = parse_coco_collect(base_dir, split, data_variant=data_variant)
        all_records.extend(records)
        print(f"  [{split}] {len(records)} records")

    for split in SPLITS:
        prepare_yolo_flat_dir(
            split,
            all_file_maps[split],
            [r for r in all_records if r["split"] == split],
            work_dir=work_dir,
        )

    data = pd.DataFrame(all_records)
    print(f"\nTotal: {data.shape}, Classes: {data['class'].value_counts().sort_index().to_dict()}")
    return data


def split_dataframes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter out the placeholder class==0 rows and split by 'split' column."""
    train_df = df[df["split"] == "train"].query("`class` > 0").copy()
    valid_df = df[df["split"] == "valid"].query("`class` > 0").copy()
    test_df = df[df["split"] == "test"].query("`class` > 0").copy()
    for name, d in [("Train", train_df), ("Val", valid_df), ("Test", test_df)]:
        print(f"{name}: {len(d)} ann, {d['file'].nunique()} imgs")
    return train_df, valid_df, test_df
