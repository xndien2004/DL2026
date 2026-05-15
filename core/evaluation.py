"""Evaluation pipeline: metrics, plots, and TP/FP/FN CSV export."""

from __future__ import annotations

import pandas as pd

from .metrics import match_dets_gts


def export_predictions_csv(model_name: str, dets, gts, image_ids,
                           iou_thr: float, score_thr: float,
                           csv_out: str, label_to_name=None) -> pd.DataFrame:
    """Write a long-format CSV with one row per TP / FP / FN.

    Returns the DataFrame for further inspection.
    """
    ltn = label_to_name or {}
    rows = []
    for img_idx, (det, gt, fid) in enumerate(zip(dets, gts, image_ids)):
        base = {"model_name": model_name, "image_id": fid}
        matches, pb, pl, ps, gb, gl, pm, gm = match_dets_gts(det, gt, iou_thr, score_thr)

        for pi, gi, iou_val in matches:
            rows.append({
                **base, "result": "TP",
                "gt_class": int(gl[gi]),
                "gt_class_name": ltn.get(int(gl[gi])),
                "gt_x1": gb[gi][0], "gt_y1": gb[gi][1],
                "gt_x2": gb[gi][2], "gt_y2": gb[gi][3],
                "pred_class": int(pl[pi]),
                "pred_class_name": ltn.get(int(pl[pi])),
                "pred_x1": pb[pi][0], "pred_y1": pb[pi][1],
                "pred_x2": pb[pi][2], "pred_y2": pb[pi][3],
                "pred_score": float(ps[pi]), "iou": iou_val,
            })
        for pi in range(len(pl)):
            if pm[pi]:
                continue
            rows.append({
                **base, "result": "FP",
                "pred_class": int(pl[pi]),
                "pred_class_name": ltn.get(int(pl[pi])),
                "pred_x1": pb[pi][0], "pred_y1": pb[pi][1],
                "pred_x2": pb[pi][2], "pred_y2": pb[pi][3],
                "pred_score": float(ps[pi]),
            })
        for gi in range(len(gl)):
            if gm[gi]:
                continue
            rows.append({
                **base, "result": "FN",
                "gt_class": int(gl[gi]),
                "gt_class_name": ltn.get(int(gl[gi])),
                "gt_x1": gb[gi][0], "gt_y1": gb[gi][1],
                "gt_x2": gb[gi][2], "gt_y2": gb[gi][3],
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(csv_out, index=False)

    tp_n = int((pred_df["result"] == "TP").sum())
    fp_n = int((pred_df["result"] == "FP").sum())
    fn_n = int((pred_df["result"] == "FN").sum())
    prec = tp_n / max(tp_n + fp_n, 1)
    rec = tp_n / max(tp_n + fn_n, 1)
    print(f"Saved: {csv_out} ({len(pred_df)} rows) "
          f"TP={tp_n} FP={fp_n} FN={fn_n} P={prec:.4f} R={rec:.4f}")
    if "gt_class_name" in pred_df.columns:
        print(pred_df.groupby(["gt_class_name", "result"]).size().unstack(fill_value=0))
    return pred_df
