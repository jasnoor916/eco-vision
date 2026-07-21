"""
Grip-free live pipeline on Gretchen's camera:
  detect a recycling item in the full frame -> stable-vote over recent frames ->
  actions.trigger() (Groq advice + TTS + nod/shake, from actions.py)

Unlike main.py, this version does NOT require the item to be gripped by a hand.
It runs the item detector on every frame and reacts to whatever recyclable it
sees. Robot gestures still happen automatically in robot mode, because
actions.trigger() -> get_recycling_advice() nods/shakes based on the advice.
"""

import argparse
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

import actions
from voting import weighted_vote

load_dotenv(Path(__file__).resolve().parent / ".env")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# Custom recycling detector. Its classes are:
#   {0: 'Glass', 1: 'Metal', 2: 'Paper', 3: 'Plastic', 4: 'Waste'}
ITEM_MODEL_PATH = "models/best.pt"
TARGET_CLASSES = None       # None = detect all recycling classes above

# Detection runs inside a central region of the frame (see ROI_SCALE), which
# zooms in on the presented item and hides cluttered edges. That confidence
# boost lets us keep the threshold reasonably high to reject background junk.
ITEM_CONF = 0.35

# Region-of-interest: fraction of the frame (centered) that we actually search.
# This replaces the grip gate — it says "only react to items held up in the
# middle." Smaller = tighter/less background but you must present items centered;
# larger = more forgiving but more clutter creeps in.
ROI_SCALE = 0.45

# Stable-vote debounce: fire once VOTE_THRESHOLD of the last VOTE_WINDOW frames
# agree on a label. Tolerates flicker without demanding a perfect streak.
VOTE_WINDOW = 6
VOTE_THRESHOLD = 4

# Re-arm policy ("once per appearance"): after firing for an item, don't fire
# again until the frame has been empty for EMPTY_REARM consecutive frames, i.e.
# the item was taken away and (maybe) a new one shown. Raise it to require a
# longer clear gap; lower it to re-arm sooner.
EMPTY_REARM = 5
# ----------------------------------------------------------------------


def camera_source(value):
    """Convert numeric camera indexes to int; keep device paths as strings."""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the grip-free EcoVision pipeline with a laptop camera or Gretchen."
    )
    parser.add_argument(
        "--mode",
        choices=("laptop", "robot"),
        default="robot",
        help="Use only the laptop camera, or initialize the robot camera and motors.",
    )
    parser.add_argument(
        "--camera",
        type=camera_source,
        default=None,
        help="Camera index or path (default: 0 for laptop; ROBOT_CAMERA for robot).",
    )
    parser.add_argument(
        "--motor-port",
        default=os.getenv("ROBOT_MOTOR_PORT", "/dev/tty.usbserial-FT94ELHH"),
        help="Robot motor serial port (robot mode only).",
    )
    return parser.parse_args()


def wait_for_camera(read_frame, max_attempts=30, delay=0.2):
    """Some cameras return empty frames for the first few reads after startup.
    Poll until a real frame comes through instead of crashing on the first read."""
    for attempt in range(max_attempts):
        try:
            ret, frame = read_frame()
            if ret and frame is not None and frame.size > 0:
                print(f"Camera ready after {attempt + 1} attempt(s).")
                return
        except cv2.error:
            pass
        time.sleep(delay)
    raise RuntimeError(
        "Camera never produced a valid frame. Check the camera is "
        "physically connected and not in use by another program."
    )


def open_camera(args):
    if args.mode == "laptop":
        source = 0 if args.camera is None else args.camera
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open laptop camera: {source}")
        return capture.read, capture.release, source

    source = (
        camera_source(os.getenv("ROBOT_CAMERA", "0"))
        if args.camera is None
        else args.camera
    )
    robot = actions.start_robot(args.motor_port, source)

    def read_frame():
        ret, frame, _timestamp = robot.camera.getImage()
        return ret, frame

    return read_frame, actions.stop_robot, source


def main():
    args = parse_args()
    item_model = YOLO(ITEM_MODEL_PATH)

    read_frame, close_camera, source = open_camera(args)
    recent_preds = deque(maxlen=VOTE_WINDOW)
    armed = True            # ready to fire for the next stable item
    empty_streak = 0        # consecutive frames with no detection

    try:
        cv2.namedWindow("Gretchen Vision")
        wait_for_camera(read_frame)
        print(
            f"Running in {args.mode} mode with camera {source!r}. "
            "Press any key in the window to quit."
        )

        while True:
            ret, frame = read_frame()
            if not ret or frame is None:
                continue
            h, w = frame.shape[:2]

            # Detect only inside a centered region-of-interest. This zooms in on
            # the presented item (boosting confidence) and ignores background
            # clutter at the edges — the job the grip gate used to do.
            rx1 = int(w * (1 - ROI_SCALE) / 2)
            ry1 = int(h * (1 - ROI_SCALE) / 2)
            rx2 = w - rx1
            ry2 = h - ry1
            roi = frame[ry1:ry2, rx1:rx2]
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (180, 180, 180), 1)

            kwargs = dict(conf=ITEM_CONF, verbose=False)
            if TARGET_CLASSES is not None:
                kwargs["classes"] = TARGET_CLASSES
            item_results = item_model.predict(roi, **kwargs)[0]

            # Diagnostic: show what the model sees every frame (like main.py did).
            print(
                f"item boxes found: {len(item_results.boxes)}",
                [
                    f"{item_model.names[int(c)]} {float(p):.2f}"
                    for c, p in zip(item_results.boxes.cls, item_results.boxes.conf)
                ]
                if len(item_results.boxes)
                else "none",
            )

            label_this_frame = None
            conf_this_frame = 0.0

            if len(item_results.boxes) > 0:
                best_i = int(item_results.boxes.conf.argmax())
                cls_id = int(item_results.boxes.cls[best_i])
                conf_this_frame = float(item_results.boxes.conf[best_i])
                label_this_frame = item_model.names[cls_id]

                x1, y1, x2, y2 = item_results.boxes.xyxy[best_i].cpu().numpy()
                # Detection is in ROI coordinates — shift back into the full frame.
                bx1, by1 = rx1 + int(x1), ry1 + int(y1)
                bx2, by2 = rx1 + int(x2), ry1 + int(y2)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{label_this_frame} {conf_this_frame:.2f}",
                    (bx1, by1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # Track empty frames so the trigger can re-arm once an item is removed.
            if label_this_frame is None:
                empty_streak += 1
                if empty_streak >= EMPTY_REARM:
                    armed = True
                    recent_preds.clear()
            else:
                empty_streak = 0

            recent_preds.append((label_this_frame, conf_this_frame))

            # Fire once a label wins the vote — but only while "armed", so a
            # single held-up item triggers once per appearance, not every frame.
            if armed:
                votes = [(lbl, c) for lbl, c in recent_preds if lbl is not None]
                # Confidence-weighted: strong detections outvote weak flickers,
                # and nobody wins while two classes are still contesting.
                winner, winner_conf = weighted_vote(votes, VOTE_THRESHOLD)
                if winner is not None:
                    # In robot mode this also nods/shakes, inside actions.
                    actions.trigger(winner, winner_conf)
                    armed = False

            status = (
                f"Detecting: {label_this_frame}" if label_this_frame else "no item"
            )
            cv2.putText(
                frame,
                status,
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if label_this_frame else (0, 0, 255),
                2,
            )
            display_frame = actions.draw_text(frame, actions.display_text)
            cv2.imshow("Gretchen Vision", display_frame)
            if cv2.waitKey(1) > 0:
                break
    finally:
        close_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
