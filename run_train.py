"""Top-level entry: train YOLOv12m + SimAM (V4).

Recommended workflow:
    1. python run_baseline_train.py          # trains baseline -> output/yolov12m_base_mix/weights/best.pt
    2. python run_train.py                   # loads baseline, trains SimAM on top -> output/yolov12m_new_mix/

Examples:
    python run_train.py
    python run_train.py --skip-phase12
    python run_train.py --pretrained ./output/yolov12m_base_mix/weights/best.pt
"""
from yolov12m_simam_pipeline.cli import parse_train_args
from yolov12m_simam_pipeline.train import main

if __name__ == "__main__":
    main(**parse_train_args())
