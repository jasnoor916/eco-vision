"""
Guided dataset capture for fine-tuning models/best.pt on YOUR items and room.

Hold ONE item, press the number key for its TRUE class, then press SPACE to
save frames. The current detector proposes the bounding box (its predicted
class is ignored — you already declared the truth), so each saved frame gets a
ready-to-train YOLO label with zero manual annotation.

Keys:
  0-4    START continuously saving frames labeled with that class:
           0=Glass  1=Metal  2=Paper  3=Plastic  4=Waste
         (press the same number again to stop; a different number switches class)
  s      stop saving
  q      quit

While recording (~3 saves/sec), slowly rotate the item and move it across
different backgrounds — every save should look a little different.

Capture tips (variety is what fixes the flicker):
  - 30-80 saves per problem item
  - vary distance, angle, rotation, and how much your hand covers it
  - vary the BACKGROUND behind a transparent bottle (this is the killer:
    hold it against the ceiling, the monitor, a wall, your shirt...)
  - capture in the same room/lighting you will demo in
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ITEM_MODEL_PATH = "models/best.pt"
# Low threshold on purpose: we only need a plausible BOX, not a correct class.
BOX_PROPOSAL_CONF = 0.15

CLASS_NAMES = {0: "Glass", 1: "Metal", 2: "Paper", 3: "Plastic", 4: "Waste"}

# Seconds between automatic saves while recording. ~0.35s = ~3 images/sec:
# fast enough to build a set quickly, slow enough that each frame differs
# (30fps of near-duplicates would inflate the dataset without adding variety).
SAVE_INTERVAL = 0.35

OUT_IMAGES = Path("data/custom/images")
OUT_LABELS = Path("data/custom/labels")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture + auto-label training images.")
    parser.add_argument("--camera", type=int, default=0, help="Laptop camera index.")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    model = YOLO(ITEM_MODEL_PATH)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    current_class = 3  # start on Plastic, the main troublemaker
    recording = False
    last_save = 0.0
    saved = 0
    print(__doc__)

    try:
        while True:
            ret, frame = capture.read()
            if not ret or frame is None:
                continue
            h, w = frame.shape[:2]

            # Propose a box with the existing model. Class prediction is shown
            # only for curiosity; the saved label uses YOUR declared class.
            results = model.predict(frame, conf=BOX_PROPOSAL_CONF, verbose=False)[0]
            box = None
            display = frame.copy()
            if len(results.boxes) > 0:
                best_i = int(results.boxes.conf.argmax())
                x1, y1, x2, y2 = results.boxes.xyxy[best_i].cpu().numpy()
                box = (x1, y1, x2, y2)
                predicted = model.names[int(results.boxes.cls[best_i])]
                conf = float(results.boxes.conf[best_i])
                cv2.rectangle(
                    display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
                cv2.putText(
                    display,
                    f"box ok (model guesses {predicted} {conf:.2f})",
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            status = "REC" if recording else "paused"
            cv2.putText(
                display,
                f"[{status}] class: [{current_class}] {CLASS_NAMES[current_class]}   "
                f"saved: {saved}   (0-4 record, s stop, q quit)",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if recording else (0, 255, 255),
                2,
            )
            if box is None:
                cv2.putText(
                    display,
                    "no box proposed - adjust item until a green box appears",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Capture dataset", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4")):
                pressed = int(chr(key))
                if recording and pressed == current_class:
                    recording = False
                    print(f"Stopped recording ({CLASS_NAMES[current_class]}).")
                else:
                    current_class = pressed
                    recording = True
                    print(f"RECORDING {CLASS_NAMES[current_class]} — press "
                          f"{pressed} again or 's' to stop.")
            elif key == ord("s"):
                recording = False
                print("Stopped recording.")

            # Auto-save while recording, throttled so each frame differs.
            now = time.time()
            if recording and box is not None and (now - last_save) >= SAVE_INTERVAL:
                x1, y1, x2, y2 = box
                # YOLO label format: class cx cy bw bh, all normalized 0-1.
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                stem = f"{CLASS_NAMES[current_class].lower()}_{int(now * 1000)}"
                cv2.imwrite(str(OUT_IMAGES / f"{stem}.jpg"), frame)
                (OUT_LABELS / f"{stem}.txt").write_text(
                    f"{current_class} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                )
                saved += 1
                last_save = now
                print(f"saved #{saved}: {stem}.jpg  ({CLASS_NAMES[current_class]})")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print(f"\nDone: {saved} labeled images in {OUT_IMAGES.parent}/")
        print("Next: python finetune.py")


if __name__ == "__main__":
    main()
