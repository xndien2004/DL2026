"""Top-level entry: pseudo-label train images using a V4 checkpoint, then retrain.

Examples:
    python run_pseudo_label.py
    python run_pseudo_label.py /path/to/v4_best.pt
    python run_pseudo_label.py best.pt --pseudo-conf 0.85 --pseudo-iou 0.30
"""

from yolov12m_simam_pipeline.cli import parse_pseudo_args
from yolov12m_simam_pipeline.pseudo_label import main


if __name__ == "__main__":
    checkpoint, overrides = parse_pseudo_args()
    main(checkpoint=checkpoint, **overrides)
