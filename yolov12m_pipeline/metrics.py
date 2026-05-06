"""Advanced metrics: confusion matrix, ranking-based PR / F1 curves, AP."""

from __future__ import annotations

import numpy as np
import torch
import torchvision
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------
def iou_matrix(a, b) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return torchvision.ops.box_iou(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
    ).numpy()


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------
def collect_predictions(mdl, dataset, score_threshold: float = 0.01):
    """Run the model on every image in `dataset`, returning (detections, gts)."""
    dets, gts = [], []
    for i in tqdm(range(len(dataset)), desc="Collecting predictions", leave=False):
        img, target, _ = dataset[i]
        pred = mdl.predict(img)
        keep = pred["scores"] >= score_threshold
        dets.append({k: mdl._to_numpy(pred[k][keep]) for k in ("boxes", "labels", "scores")})
        gts.append({k: mdl._to_numpy(target[k]) for k in ("boxes", "labels")})
    return dets, gts


# ---------------------------------------------------------------------------
# Det/GT matching (used by confusion matrix + CSV export)
# ---------------------------------------------------------------------------
def match_dets_gts(det, gt, iou_thr: float, score_thr: float):
    """Return (matches, pb, pl, ps, gb, gl, pred_matched, gt_matched)."""
    keep = det["scores"] >= score_thr
    pb, pl, ps = det["boxes"][keep], det["labels"][keep].astype(int), det["scores"][keep]
    gb, gl = gt["boxes"], gt["labels"].astype(int)
    gm = np.zeros(len(gl), dtype=bool)
    pm = np.zeros(len(pl), dtype=bool)
    matches = []
    if len(pb) > 0 and len(gb) > 0:
        iou = iou_matrix(pb, gb)
        for pi in np.argsort(-ps):
            gi = np.argmax(iou[pi])
            if iou[pi, gi] >= iou_thr and not gm[gi]:
                matches.append((pi, gi, float(iou[pi, gi])))
                pm[pi] = gm[gi] = True
    return matches, pb, pl, ps, gb, gl, pm, gm


def build_confusion_matrix(dets, gts, num_cls: int,
                           iou_thr: float = 0.5, score_thr: float = 0.25) -> np.ndarray:
    n = num_cls + 1
    cm = np.zeros((n, n), dtype=int)
    for det, gt in zip(dets, gts):
        matches, pb, pl, ps, gb, gl, pm, gm = match_dets_gts(det, gt, iou_thr, score_thr)
        for pi, gi, _ in matches:
            pc, gc = int(pl[pi]) - 1, int(gl[gi]) - 1
            if 0 <= pc < num_cls and 0 <= gc < num_cls:
                cm[gc, pc] += 1
        for gi, m in enumerate(gm):
            if not m:
                gc = int(gl[gi]) - 1
                if 0 <= gc < num_cls:
                    cm[gc, num_cls] += 1
        for pi, m in enumerate(pm):
            if not m:
                pc = int(pl[pi]) - 1
                if 0 <= pc < num_cls:
                    cm[num_cls, pc] += 1
    return cm


# ---------------------------------------------------------------------------
# PR curve (ranking-based, Ultralytics style)
# ---------------------------------------------------------------------------
def pr_curve_ranking(dets, gts, cls: int, iou_thr: float = 0.5):
    """Sort all detections by score, accumulate TP/FP. Returns (precision, recall, scores)."""
    total_gt = sum((g["labels"] == cls).sum() for g in gts)
    if total_gt == 0:
        return np.array([1.0]), np.array([0.0]), np.array([0.0])

    all_dets = []
    for img_idx, det in enumerate(dets):
        mask = det["labels"] == cls
        for j in np.where(mask)[0]:
            all_dets.append({"score": det["scores"][j], "box": det["boxes"][j], "img": img_idx})
    if not all_dets:
        return np.array([1.0]), np.array([0.0]), np.array([0.0])

    all_dets.sort(key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(all_dets))
    fp = np.zeros(len(all_dets))
    matched: dict[int, set] = {i: set() for i in range(len(dets))}

    for i, d in enumerate(all_dets):
        gt = gts[d["img"]]
        gt_mask = gt["labels"] == cls
        gt_boxes = gt["boxes"][gt_mask]
        gt_ids = np.where(gt_mask)[0]

        if len(gt_boxes) == 0:
            fp[i] = 1
            continue

        iou = iou_matrix(d["box"][np.newaxis], gt_boxes)[0]
        best_gt = np.argmax(iou)
        if iou[best_gt] >= iou_thr and gt_ids[best_gt] not in matched[d["img"]]:
            tp[i] = 1
            matched[d["img"]].add(gt_ids[best_gt])
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum)
    scores = np.array([d["score"] for d in all_dets])
    return precision, recall, scores


def smooth_pr_ultralytics(precision, recall, num_points: int = 1000):
    """Monotonic envelope over a uniform recall grid [0, 1]."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    r_grid = np.linspace(0, 1, num_points)
    p_grid = np.interp(r_grid, mrec, mpre)
    return p_grid, r_grid


def compute_ap(precision, recall) -> float:
    """All-point interpolation, integrate over 101-point grid."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def f1_from_ranking(dets, gts, cls: int, iou_thr: float = 0.5, num_points: int = 1000):
    """F1 vs. confidence threshold, derived from the ranking PR curve."""
    prec, rec, scores = pr_curve_ranking(dets, gts, cls, iou_thr)
    conf_grid = np.linspace(0, 1, num_points)

    if len(scores) <= 1:
        return np.zeros(num_points), conf_grid

    prec_ext = np.concatenate(([1.0], prec))
    rec_ext = np.concatenate(([0.0], rec))
    n = np.searchsorted(-scores, -conf_grid)
    n = np.clip(n, 0, len(prec))
    p_at_conf = prec_ext[n]
    r_at_conf = rec_ext[n]
    f1 = 2 * p_at_conf * r_at_conf / np.maximum(p_at_conf + r_at_conf, 1e-16)
    return f1, conf_grid
