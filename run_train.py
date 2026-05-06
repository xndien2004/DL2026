"""Top-level entry: train YOLOv12m + SimAM (V4).

Examples:
    python run_train.py
    python run_train.py --skip-phase12
    python run_train.py --base-dir ./data/DataAug --pretrained ./weight/best.pt
    python run_train.py --custom-arch simam --variant mix
"""

from yolov12m_simam_pipeline.cli import parse_train_args
from yolov12m_simam_pipeline.train import main


if __name__ == "__main__":
    main(**parse_train_args())
