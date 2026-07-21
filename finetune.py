"""
Fine-tune the existing recycling detector on your newly captured images.

Unlike train.py (which builds a model from scratch), this starts from
models/best.pt so everything it already knows is kept, and trains on the
ORIGINAL dataset PLUS data/custom (see data/data_finetune.yaml). The custom
images teach it your camera, lighting, hands, and specific items.

Workflow:
  1. python capture_dataset.py          # collect + auto-label your images
  2. python finetune.py                 # this script
  3. copy the printed best.pt over models/best.pt (keep a backup first!)
"""

from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data" / "data_finetune.yaml"
BASE_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"


def pick_device():
    """CUDA on an NVIDIA box, MPS on Apple Silicon, CPU everywhere else."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    n_custom = len(list((PROJECT_ROOT / "data" / "custom" / "images").glob("*.jpg")))
    if n_custom == 0:
        raise SystemExit(
            "No captured images found in data/custom/images.\n"
            "Run `python capture_dataset.py` first."
        )
    print(f"Fine-tuning with {n_custom} custom images on top of the original set.")

    model = YOLO(str(BASE_WEIGHTS))  # start from what it already knows
    results = model.train(
        data=str(DATA),
        epochs=40,          # fine-tuning needs far fewer epochs than scratch
        imgsz=640,
        device=pick_device(),
        patience=15,        # stop early once val metrics stop improving
        lr0=0.001,          # gentler learning rate: nudge weights, don't stomp them
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print(f"New weights: {best}")
    print("To deploy (after backing up the old model):")
    print("  cp models/best.pt models/best_backup.pt")
    print(f"  cp '{best}' models/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
