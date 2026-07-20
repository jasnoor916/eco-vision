"""
Live pipeline on Gretchen's own camera:
  hand landmarks -> grip check -> item detection (only when gripped) ->
  actions.trigger() (Groq advice + TTS + nod/shake, from actions.py)
"""

import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

import actions

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
HAND_MODEL_PATH = r"C:\Users\Jun Ci\workspace\gretchen\gretchen_project\eco-vision\YOLO11n-pose-hands\runs\pose\train\weights\best.pt"  # from your friend's fine-tuning

# Start with the existing YOLOv8n bottle/cup detector (matches suggest.py).
# Swap to your custom Plastic/Paper/General model once trained.
ITEM_MODEL_PATH = r"C:\Users\Jun Ci\workspace\gretchen\gretchen_project\eco-vision\best.pt"
TARGET_CLASSES = [39, 41]  # COCO: 39=bottle, 41=cup — set to None to use all classes

HAND_CONF = 0.5
ITEM_CONF = 0.25
CROP_MARGIN = 0.6          # expand hand bbox by 60% each side before item-detection crop
GRIP_SPREAD_RATIO = 1.15    
STABLE_FRAMES = 8          # consecutive frames the same label must win before triggering
# ----------------------------------------------------------------------


def is_gripping(kpts_xy: np.ndarray) -> bool:
    """kpts_xy: (21, 2) array of pixel coordinates for one hand.

    Compares how far apart the fingertips are (index tip to pinky tip)
    against palm width (index knuckle to pinky knuckle). An open flat hand
    has fingertips spread much wider than the palm; a closed fist OR a
    hand wrapped around an object both bring the fingertips close together
    relative to palm width — which is exactly the "gripping something"
    signal we want, without over-penalizing a grip that isn't fully closed
    because there's an object in the way.
    """
    palm_width = np.linalg.norm(kpts_xy[5] - kpts_xy[17])   # index MCP <-> pinky MCP
    tip_spread = np.linalg.norm(kpts_xy[8] - kpts_xy[20])   # index tip <-> pinky tip

    if palm_width < 1e-6:
        return False

    return (tip_spread / palm_width) < GRIP_SPREAD_RATIO


def expand_box(x1, y1, x2, y2, margin, frame_w, frame_h):
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - w * margin))
    y1 = max(0, int(y1 - h * margin))
    x2 = min(frame_w, int(x2 + w * margin))
    y2 = min(frame_h, int(y2 + h * margin))
    return x1, y1, x2, y2

def wait_for_camera(robot, max_attempts=30, delay=0.2):
    """Some cameras return empty frames for the first few reads after
    startup. Poll until a real frame comes through instead of crashing
    on the first getImage() call."""
    for attempt in range(max_attempts):
        try:
            ret, frame, timestamp = robot.camera.getImage()
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

def main():
    hand_model = YOLO(HAND_MODEL_PATH)
    item_model = YOLO(ITEM_MODEL_PATH)

    cv2.namedWindow("Gretchen Vision")
    wait_for_camera(actions.robot)

    recent_labels = deque(maxlen=STABLE_FRAMES)
    already_triggered_this_grip = False

    print("Running. Press any key in the window to quit.")
    while True:
        ret, frame, timestamp = actions.robot.camera.getImage()
        if not ret:
            continue
        h, w = frame.shape[:2]

        hand_results = hand_model.predict(frame, conf=HAND_CONF, verbose=False)[0]

        grip_active = False
        label_this_frame = None
        conf_this_frame = 0.0

        if hand_results.keypoints is not None and len(hand_results.boxes) > 0:
            best_idx = int(hand_results.boxes.conf.argmax())
            box = hand_results.boxes.xyxy[best_idx].cpu().numpy()
            kpts = hand_results.keypoints.xy[best_idx].cpu().numpy()  # (21, 2)

            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
            for px, py in kpts:
                cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 255), -1)

            if is_gripping(kpts):
                grip_active = True
                cx1, cy1, cx2, cy2 = expand_box(x1, y1, x2, y2, CROP_MARGIN, w, h)
                crop = frame[cy1:cy2, cx1:cx2]

                if crop.size > 0:
                    # Ensure crop is a proper format for YOLO
                    crop_processed = crop.copy()  # Make a proper copy
                    if crop_processed.dtype != np.uint8:
                        crop_processed = crop_processed.astype(np.uint8)
                    
                    kwargs = dict(conf=ITEM_CONF, verbose=False)
                    if TARGET_CLASSES is not None:
                        kwargs["classes"] = TARGET_CLASSES
                    
                    try:
                        # Try prediction with processed crop
                        item_results = item_model.predict(crop_processed, **kwargs)[0]
                        print(f"item boxes found: {len(item_results.boxes)}", 
                            [item_model.names[int(c)] for c in item_results.boxes.cls] if len(item_results.boxes) else "none")
                        
                        if len(item_results.boxes) > 0:
                            best_i = int(item_results.boxes.conf.argmax())
                            cls_id = int(item_results.boxes.cls[best_i])
                            conf_this_frame = float(item_results.boxes.conf[best_i])
                            label_this_frame = item_model.names[cls_id]

                    if len(item_results.boxes) > 0:
                        best_i = int(item_results.boxes.conf.argmax())
                        cls_id = int(item_results.boxes.cls[best_i])
                        conf_this_frame = float(item_results.boxes.conf[best_i])
                        label_this_frame = item_model.names[cls_id]

                        ix1, iy1, ix2, iy2 = item_results.boxes.xyxy[best_i].cpu().numpy()
                        cv2.rectangle(
                            frame,
                            (cx1 + int(ix1), cy1 + int(iy1)),
                            (cx1 + int(ix2), cy1 + int(iy2)),
                            (0, 0, 255), 2,
                        )
                        cv2.putText(
                            frame,
                            f"{label_this_frame} {conf_this_frame:.2f}",
                            (cx1 + int(ix1), cy1 + int(iy1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                        )

                        palm_width = np.linalg.norm(kpts[5] - kpts[17])
                        tip_spread = np.linalg.norm(kpts[8] - kpts[20])
                        print(f"spread ratio: {tip_spread/palm_width:.2f}")

        # --- debounce / trigger logic ---
        if grip_active and label_this_frame is not None:
            recent_labels.append(label_this_frame)
        else:
            recent_labels.clear()
            already_triggered_this_grip = False

        if (
            len(recent_labels) == STABLE_FRAMES
            and len(set(recent_labels)) == 1
            and not already_triggered_this_grip
        ):
            actions.trigger(recent_labels[-1], conf_this_frame)
            already_triggered_this_grip = True  # wait for grip release before firing again

        cv2.putText(
            frame,
            "GRIP" if grip_active else "no grip",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0) if grip_active else (0, 0, 255), 2,
        )
        display_frame = actions.draw_text(frame, actions.display_text)
        cv2.imshow("Gretchen Vision", display_frame)
        if cv2.waitKey(1) > 0:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()