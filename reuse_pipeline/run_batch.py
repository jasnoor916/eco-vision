"""
Batch-runs the reuse pipeline over a handful of images from data/test/images.
Useful for sanity-checking multiple category/condition combinations at once --
running only test_image.jpg only ever exercises one code path (e.g. you already
saw Plastic/damaged -> correctly withheld; this will surface an intact item
getting a Groq-phrased suggestion, contamination cases, etc.)

Usage (run from inside reuse_pipeline/):
  python3 run_batch.py                          # samples 5 images automatically
  python3 run_batch.py --n 8                    # samples 8 images
  python3 run_batch.py --images 000000_jpg.rf.0cca9fadb7533c08488f934280a8ade3.jpg
                                                  # run specific file(s) by name

Writes one JSON file per image into runs/<timestamp>/<image_stem>.json and
prints a one-line summary per detected object as it goes. Respects the
existing MAX_API_CALLS_PER_RUN budget in waste_pipeline.py -- if the budget
is hit partway through, images already completed are still saved.
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

import waste_pipeline as wp

TEST_IMAGES_DIR = Path("../data/test/images")


def pick_images(n: int) -> list[Path]:
    all_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not all_images:
        sys.exit(f"No .jpg files found in {TEST_IMAGES_DIR.resolve()}")
    random.seed(42)  # reproducible sample across runs
    return random.sample(all_images, min(n, len(all_images)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5,
                         help="number of images to sample if --images not given")
    parser.add_argument("--images", nargs="*",
                         help="specific filenames in data/test/images to run instead of sampling")
    args = parser.parse_args()

    if args.images:
        images = [TEST_IMAGES_DIR / name for name in args.images]
        missing = [p for p in images if not p.exists()]
        if missing:
            sys.exit(f"Not found: {missing}")
    else:
        images = pick_images(args.n)

    out_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo_model = YOLO(wp.YOLO_MODEL_PATH)
    kb = wp.load_knowledge_base(wp.KNOWLEDGE_BASE_PATH)

    print(f"Running pipeline on {len(images)} images -> {out_dir}/\n")

    for img_path in images:
        print(f"--- {img_path.name} ---")
        try:
            result = wp.process_image(str(img_path), yolo_model, kb)
        except wp.ApiCallBudgetExceeded as e:
            print(f"[STOPPED] {e}")
            break

        out_file = out_dir / f"{img_path.stem}.json"
        out_file.write_text(json.dumps(result, indent=2))

        if not result["objects"]:
            print("  (no objects detected)")
        for obj in result["objects"]:
            reuse = obj["reuse"]

            print(f"\n{obj['category']} ({obj['condition']['status']})")

            if reuse["eligible"]:
                print("✅ Reuse Possible")

                if reuse.get("title"):
                    print(f"\n{reuse['title']}")

                if reuse.get("why"):
                    print(reuse["why"])

                if reuse.get("steps"):
                    print("\nIdeas:")
                    for step in reuse["steps"]:
                        print(f"  • {step}")

                if reuse.get("source_citation"):
                    print(f"\nSource: {reuse['source_citation']}")

            else:
                print("❌ Not suitable for reuse")
                if reuse.get("reason"):
                    print(reuse["reason"])

print()
print(f"Total provider calls this run: {wp._api_call_count}/{wp.MAX_API_CALLS_PER_RUN}")


if __name__ == "__main__":
    main()