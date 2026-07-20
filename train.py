from pathlib import Path

import torch
from ultralytics import YOLO

# data.yaml deliberately omits a `path:` key, so Ultralytics resolves the split
# folders relative to the YAML itself — this works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data" / "data.yaml"


def pick_device():
    """CUDA on an NVIDIA box, MPS on Apple Silicon, CPU everywhere else."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    model_cfg = "yolo26-p2.yaml"

    model = YOLO(model_cfg)

    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=640,
        device=pick_device(),
    )


if __name__ == "__main__":
    main()
