"""Top-level entry: train the baseline yolov12m pipeline (no SimAM).

Examples:
    python run_baseline_train.py
    python run_baseline_train.py --epochs 100 --batch 32
    python run_baseline_train.py --base-dir ./data/DataAug --variant bright
"""

from baseline.cli import parse_train_args
from baseline.train import main


if __name__ == "__main__":
    main(**parse_train_args())
