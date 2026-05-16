"""Shared argparse helpers for pipeline CLI modules."""
from __future__ import annotations
import argparse


def _add_common(parser: argparse.ArgumentParser, flags: dict[str, str]) -> None:
    add = parser.add_argument
    if "model" in flags:        add("--model",        type=str,   help="Model name (yolov12m, yolov8s, rtdetr-l, ...)")
    if "epochs" in flags:       add("--epochs",       type=int,   help="Number of training epochs")
    if "lr" in flags:           add("--lr",           type=float, help="Initial learning rate")
    if "imgsz" in flags:        add("--imgsz",        type=int,   help="Training/eval image size")
    if "batch" in flags:        add("--batch",        type=int,   help="Ultralytics trainer batch size")
    if "loader_batch" in flags: add("--loader-batch", dest="loader_batch", type=int, help="DataLoader batch size")
    if "patience" in flags:     add("--patience",     type=int,   help="Early-stopping patience")
    if "base_dir" in flags:     add("--base-dir",     dest="base_dir", type=str, help="Source dataset root")
    if "work_dir" in flags:     add("--work-dir",     dest="work_dir", type=str, help="YOLO-flat working dir")
    if "variant" in flags:      add("--variant",      type=str, choices=["mix", "bright", "dark"], help="Lighting subset")
    if "iou" in flags:          add("--iou",          type=float, help="IoU threshold")
    if "score" in flags:        add("--score",        type=float, help="Score threshold")
    if "n_examples" in flags:   add("--n-examples",   dest="n_examples", type=int, help="Number of test images to render")
    if "pretrained" in flags:   add("--pretrained",   type=str,   help="Pretrained .pt checkpoint to fine-tune from")
    if "custom_arch" in flags:  add("--custom-arch",  dest="custom_arch", type=str, choices=["", "simam"])
    if "skip_phase12" in flags: add("--skip-phase12", dest="skip_phase12", action="store_true", default=None)


def _to_overrides(args: argparse.Namespace, flags: dict[str, str]) -> dict:
    out: dict = {}
    for cli_name, cfg_name in flags.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            out[cfg_name] = val
    return out
