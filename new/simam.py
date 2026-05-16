"""SimAM (Yang et al. 2021) — parameter-FREE attention.

Computes attention weights from each neuron's energy function, no learnable
parameters. Injected into ``ultralytics.nn.tasks`` so DDP children processes
can resolve the symbol after a fork as well.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """SimAM - parameter-free attention via neuron energy function."""

    def __init__(self, c1, *args, e_lambda: float = 1e-4, **kwargs):
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_sq / (
            4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / max(n, 1) + self.e_lambda)
        ) + 0.5
        return x * self.activation(y)


_SIMAM_SRC = """

# === CUSTOM_SIMAM_INJECTED_v4 ===
import torch as _torch_simam
import torch.nn as _nn_simam

class SimAM(_nn_simam.Module):
    def __init__(self, c1, *args, e_lambda=1e-4, **kwargs):
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = _nn_simam.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_sq / (
            4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / max(n, 1) + self.e_lambda)
        ) + 0.5
        return x * self.activation(y)
"""


def inject_simam() -> None:
    """Register SimAM in the parent ultralytics process and append the source
    to ``ultralytics/nn/tasks.py`` so DDP-spawned worker processes also see it.
    """
    import ultralytics.nn.tasks as _tasks
    _tasks.SimAM = SimAM

    tasks_file = _tasks.__file__
    with open(tasks_file, "r") as f:
        content = f.read()
    if "CUSTOM_SIMAM_INJECTED_v4" not in content:
        with open(tasks_file, "a") as f:
            f.write(_SIMAM_SRC)
        print(f"[DDP-safe] Injected SimAM into {tasks_file}")
    else:
        print(f"[DDP-safe] SimAM already injected in {tasks_file}")
    print("SimAM module registered (parent + DDP children) - 0 params!")


# ---------------------------------------------------------------------------
# Custom YOLOv12 + SimAM YAML builder
# ---------------------------------------------------------------------------
_SCALES = {
    "n": [0.50, 0.25, 1024], "s": [0.50, 0.50, 1024],
    "m": [0.50, 1.00, 512],  "l": [1.00, 1.00, 512], "x": [1.00, 1.50, 512],
}

_SIMAM_CHANNELS = {
    "n": {"p3":  64, "p4": 128, "p5": 256},
    "s": {"p3": 128, "p4": 256, "p5": 512},
    "m": {"p3": 256, "p4": 512, "p5": 512},
    "l": {"p3": 256, "p4": 512, "p5": 512},
    "x": {"p3": 384, "p4": 512, "p5": 512},
}


def build_yolov12_simam_yaml(nc: int, base_size: str = "m") -> str:
    """Return the YAML text for YOLOv12{base_size} + SimAM at P3/P4/P5."""
    scale = _SCALES[base_size]
    ch = _SIMAM_CHANNELS[base_size]

    return f"""# YOLOv12{base_size} + SimAM (parameter-FREE attention, 0 new params)
nc: {nc}
scales:
  {base_size}: {scale}

backbone:
  - [-1, 1, Conv, [64, 3, 2]]                    # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]                   # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]            # 2
  - [-1, 1, Conv, [256, 3, 2]]                   # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]            # 4
  - [-1, 1, Conv, [512, 3, 2]]                   # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]               # 6
  - [-1, 1, Conv, [1024, 3, 2]]                  # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]              # 8

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 9
  - [[-1, 6], 1, Concat, [1]]                    # 10
  - [-1, 2, A2C2f, [512, False, -1]]             # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 12
  - [[-1, 4], 1, Concat, [1]]                    # 13
  - [-1, 2, A2C2f, [256, False, -1]]             # 14  (P3 raw)

  - [-1, 1, Conv, [256, 3, 2]]                   # 15
  - [[-1, 11], 1, Concat, [1]]                   # 16
  - [-1, 2, A2C2f, [512, False, -1]]             # 17  (P4 raw)

  - [-1, 1, Conv, [512, 3, 2]]                   # 18
  - [[-1, 8], 1, Concat, [1]]                    # 19
  - [-1, 2, C3k2, [1024, True]]                  # 20  (P5)

  - [14, 1, SimAM, [{ch["p3"]}]]                 # 21
  - [17, 1, SimAM, [{ch["p4"]}]]                 # 22
  - [20, 1, SimAM, [{ch["p5"]}]]                 # 23

  - [[21, 22, 23], 1, Detect, [nc]]              # 24
"""


def create_custom_model_yaml(arch: str | None, nc: int,
                              base_size: str = "m", out_dir=None) -> str | None:
    """Write the SimAM YAML to ``out_dir`` (or cwd) and return the path."""
    from pathlib import Path

    if not arch:
        return None
    if arch != "simam":
        raise ValueError(f"Unknown CUSTOM_ARCH='{arch}'. Supported: 'simam'.")

    yaml_content = build_yolov12_simam_yaml(nc, base_size)
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = str(out_dir / f"yolo12{base_size}-simam.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created custom architecture YAML: {yaml_path}")
    return yaml_path
