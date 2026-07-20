"""Run the trained recycling detector on a sample of test images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "test" / "images"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "test_predictions"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw predicted bounding boxes on a sample of test images."
    )
    parser.add_argument("--count", type=int, default=10, help="number of images to use")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--seed", type=int, default=42, help="random sampling seed")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="model weights")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="image folder")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.source.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {args.source}")
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    images = sorted(
        path
        for path in args.source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {args.source}")

    selected = random.Random(args.seed).sample(images, min(args.count, len(images)))
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    results = model.predict(
        source=[str(path) for path in selected],
        conf=args.conf,
        save=False,
        verbose=False,
    )

    print(f"Processed {len(results)} image(s):")
    for source_path, result in zip(selected, results, strict=True):
        output_path = args.output / source_path.name
        result.save(filename=str(output_path))
        detection_count = 0 if result.boxes is None else len(result.boxes)
        print(f"  {output_path.name}: {detection_count} detection(s)")

    print(f"Annotated images saved to: {args.output}")


if __name__ == "__main__":
    main()
