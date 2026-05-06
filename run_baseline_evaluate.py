"""Top-level entry: evaluate a baseline yolov12m checkpoint.

Examples:
    python run_baseline_evaluate.py
    python run_baseline_evaluate.py /path/to/best.pt
    python run_baseline_evaluate.py best.pt --iou 0.5 --score 0.25 --n-examples 50
"""

from yolov12m_pipeline.cli import parse_eval_args
from yolov12m_pipeline.evaluate import main


if __name__ == "__main__":
    checkpoint, overrides = parse_eval_args()
    main(checkpoint=checkpoint, **overrides)
