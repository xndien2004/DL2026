# Drill Defect Detection — Baseline + V4 (SimAM)

Two pipelines side-by-side in this directory:

| Pipeline       | Package                       | Bash driver                       | Description                                  |
|----------------|-------------------------------|-----------------------------------|----------------------------------------------|
| Baseline       | `yolov12m_pipeline/`          | `baseline.sh`                     | Stock YOLOv12m, single-stage train + eval    |
| V4 / SimAM     | `yolov12m_simam_pipeline/`    | `change_architecture_model.sh`    | YOLOv12m + parameter-free SimAM, 3-phase + optional pseudo-label retrain |

Converted from `yolov12m-baseline.ipynb` (baseline) and `newMethod_v4.ipynb`
(SimAM V4).

## Layout

```
source1/
├── baseline.sh                       # driver for the baseline pipeline
├── change_architecture_model.sh      # driver for the SimAM V4 pipeline
│
├── run_baseline_train.py             # baseline entry: train
├── run_baseline_evaluate.py          # baseline entry: evaluate
│
├── run_train.py                      # V4 entry: train (3-phase)
├── run_evaluate.py                   # V4 entry: evaluate
├── run_pseudo_label.py               # V4 entry: pseudo-label retrain
│
├── download_data.py                  # HuggingFace dataset downloader
├── requirements.txt
│
├── yolov12m_pipeline/                # baseline package (from source/)
│   ├── config.py / cli.py
│   ├── data_preparation.py / dataset.py / visualization.py
│   ├── detection_base.py / ultralytics_detector.py
│   ├── metrics.py / plots.py / evaluation.py
│   └── train.py / evaluate.py
│
└── yolov12m_simam_pipeline/          # V4 package (newMethod_v4)
    ├── config.py / cli.py
    ├── data_preparation.py / dataset.py / visualization.py
    ├── detection_base.py / ultralytics_detector.py
    ├── simam.py                      # SimAM module + custom YAML builder
    ├── metrics.py / plots.py / evaluation.py
    ├── train.py                      # 3-phase training
    ├── evaluate.py
    └── pseudo_label.py               # generate pseudo-labels + retrain
```

## Expected directory layout (relative to the bash drivers)

```
./data/DataAug      <- dataset (auto-downloaded from HuggingFace if missing)
./weight/best.pt    <- pretrained checkpoint (V4 fine-tunes from this)
./yolo_dataset      <- generated YOLO-flat layout
./runs/...          <- V4 ultralytics outputs
./detection_runs/.. <- baseline ultralytics outputs
```

The dataset is fetched from
[`nhonhoccode/DetectDrill`](https://huggingface.co/datasets/nhonhoccode/DetectDrill).

## Quick start

Both bash drivers take two positional args:

```
bash <driver>.sh [MODE] [VARIANT]
  MODE     full | skip-install | eval-only [| pseudo]   default: full
  VARIANT  mix  | bright | dark | all                    default: mix
```

Each variant gets its own `yolo_dataset_<variant>/` and
`{detection_,}runs/<run-name>_<variant>/` directory so multiple lighting
subsets coexist without clobbering each other. `all` iterates `mix → bright → dark`.

### Baseline (yolov12m only)

```bash
bash baseline.sh                       # full + mix (default)
bash baseline.sh full bright
bash baseline.sh full dark
bash baseline.sh full all              # train+eval on mix, bright, dark sequentially
bash baseline.sh skip-install bright   # skip pip install
bash baseline.sh eval-only mix         # only run evaluation
```

### V4 / SimAM (with optional pseudo-label retrain)

```bash
bash change_architecture_model.sh                       # full + mix
bash change_architecture_model.sh full bright
bash change_architecture_model.sh full all              # iterate all 3 variants
bash change_architecture_model.sh skip-install dark
bash change_architecture_model.sh eval-only mix
bash change_architecture_model.sh pseudo bright         # full + pseudo-label retrain
```

## Manual invocations

```bash
# --- Baseline ---
python run_baseline_train.py --base-dir ./data/DataAug
python run_baseline_evaluate.py detection_runs/yolov12m/weights/best.pt \
    --base-dir ./data/DataAug

# --- V4 ---
python run_train.py --base-dir ./data/DataAug --pretrained ./weight/best.pt
python run_evaluate.py runs/yolov12m-simam/weights/best.pt \
    --base-dir ./data/DataAug --pretrained ./weight/best.pt
python run_pseudo_label.py runs/yolov12m-simam/weights/best.pt \
    --base-dir ./data/DataAug --pseudo-conf 0.85
```

## Notable defaults (V4 — override via CLI flags)

| Flag             | Config attr          | Default                              |
|------------------|----------------------|--------------------------------------|
| `--model`        | `MODEL_NAME`         | `yolov12m`                           |
| `--custom-arch`  | `CUSTOM_ARCH`        | `simam`                              |
| `--epochs`       | `NUM_EPOCHS`         | `75`                                 |
| `--lr`           | `LEARNING_RATE`      | `0.001`                              |
| `--imgsz`        | `IMGSZ`              | `640` (Phase 3 uses `768`)           |
| `--batch`        | `BATCH_SIZE_YOLO`    | `24`                                 |
| `--patience`     | `PATIENCE`           | `15`                                 |
| `--iou`          | `IOU_THRESHOLD`      | `0.5`                                |
| `--score`        | `SCORE_THRESHOLD`    | `0.15`                               |
| `--variant`      | `DATA_VARIANT`       | `mix` (also: `bright`, `dark`)       |
| `--pretrained`   | `PRETRAINED_CKPT`    | `./weight/best.pt`                   |
| `--skip-phase12` | `SKIP_PHASE_1_2`     | `True` (already-converged checkpoint)|
| `--pseudo-conf`  | `PSEUDO_CONF`        | `0.85`                               |
| `--pseudo-iou`   | `PSEUDO_IOU_OVERLAP` | `0.30`                               |

## Notes

- Multi-GPU is auto-detected (`torch.cuda.device_count()`); the trainer uses
  `device="0,1"` when ≥ 2 GPUs are available.
- `SimAM` is parameter-free: the custom YAML adds it at indices 21/22/23 and
  re-routes `Detect` to consume the refined feature maps. Indices 0..20 are
  identical to stock `yolo12m.yaml`, so a stock V4 `best.pt` loads fully.
- `SimAM` is also appended to `ultralytics/nn/tasks.py` so DDP-spawned worker
  processes resolve the symbol after a `fork`.
- Baseline uses split name `valid/`; V4 uses `val/`. Both packages auto-resolve
  whichever is present on disk.
