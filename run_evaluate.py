"""Top-level entry: evaluate a trained checkpoint.

Examples:
    python run_evaluate.py
    python run_evaluate.py /path/to/best.pt
    python run_evaluate.py best.pt --iou 0.5 --score 0.15 --n-examples 50
    python run_evaluate.py best.pt --base-dir ./data/DataAug --work-dir ./yolo_dataset
"""

from yolov12m_simam_pipeline.cli import parse_eval_args
from yolov12m_simam_pipeline.evaluate import main


if __name__ == "__main__":
    checkpoint, overrides = parse_eval_args()
    main(checkpoint=checkpoint, **overrides)
