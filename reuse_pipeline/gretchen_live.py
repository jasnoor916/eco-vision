"""
Gretchen live integration for EcoVision.

Custom waste detector (recycling_model.pt) -> Gemini/NIM condition assessment
-> curated reuse KB -> Groq phrasing -> nod/shake gesture + on-screen readout.

Run from inside reuse_pipeline/ (same cwd waste_pipeline.py expects, so its
relative paths for YOLO_MODEL_PATH / KNOWLEDGE_BASE_PATH resolve correctly).
"""
import time

import cv2
from PIL import Image
from ultralytics import YOLO

from gretchen.robot import Robot

import waste_pipeline as wp

# --- Config ---
SERIAL_PORT = '/dev/tty.usbserial-FT94EO15'
LIVE_CONF_THRESHOLD = 0.55   # stricter than wp.YOLO_CONF_THRESHOLD (0.25):
                              # that low bar is fine for batch stills where
                              # condition/reuse gating happens later, but live
                              # video has motion blur/false positives we want
                              # to filter BEFORE spending an API call on them.
COOLDOWN_SECONDS = 15         # per-category pause before re-triggering the
                               # full pipeline (each trigger = up to 2 calls,
                               # see MAX_API_CALLS_PER_RUN in waste_pipeline.py)


# --- Gestures ---
def nod(robot):
    for _ in range(3):
        robot.up(); time.sleep(0.3)
        robot.down(); time.sleep(0.3)
    robot.up(); time.sleep(0.2)


def shake(robot):
    for _ in range(3):
        robot.left(); time.sleep(0.3)
        robot.right(); time.sleep(0.3)
    robot.left(); time.sleep(0.2)


def _parse_detections(results) -> list[dict]:
    """Same output shape as wp.run_detection, built from an in-memory
    results object instead of re-reading a file path."""
    detections = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls.item())
        category = results.names[cls_id]
        conf = float(box.conf.item())
        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
        detections.append({
            "object_id": f"obj_{i:03d}",
            "bbox": xyxy,
            "category": category,
            "confidence": conf,
        })
    return detections


def handle_object(det, pil_frame, kb, robot):
    """Runs the full condition + reuse pipeline on one detected object,
    reacts physically, and returns the two lines to display on-screen."""
    crop = wp.crop_object(pil_frame, det["bbox"])
    condition = wp.assess_condition(crop, det["category"])
    retrieved = wp.retrieve_reuse_options(kb, det["category"], condition.status)
    reuse = wp.phrase_reuse_suggestion(det["category"], condition.status, retrieved)

    condition_line = f"{det['category']} - condition: {condition.status}"

    if reuse.eligible:
        suggestion_line = f"Suggested use: {reuse.title} - {reuse.why}"
        steps_lines = [f"  {i+1}. {s}" for i, s in enumerate(reuse.steps or [])]
    else:
        suggestion_line = f"Not reuse-eligible: {reuse.reason}"
        steps_lines = []

    print(f"\n[Gretchen] {condition_line} "
          f"({condition.provider_used}, conf {condition.assessment_confidence:.2f})")
    print(f"[Gretchen] {suggestion_line}")
    for line in steps_lines:
        print(line)

    if reuse.eligible:
        nod(robot)
    else:
        shake(robot)

    return condition_line, suggestion_line


def _draw_overlay(frame, condition_line, suggestion_line):
    """Draws the last known condition + suggestion on the live video feed."""
    if condition_line:
        cv2.putText(frame, condition_line, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if suggestion_line:
        # Wrap long suggestion text across two lines so it doesn't run off-screen
        max_len = 60
        line1, line2 = suggestion_line[:max_len], suggestion_line[max_len:max_len*2]
        cv2.putText(frame, line1, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if line2:
            cv2.putText(frame, line2, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return frame


def main():
    robot = Robot(SERIAL_PORT, 0)
    camera = robot.camera

    yolo_model = YOLO(wp.YOLO_MODEL_PATH)
    kb = wp.load_knowledge_base(wp.KNOWLEDGE_BASE_PATH)

    last_detected = None
    last_call_time = 0.0
    last_condition_line = ""
    last_suggestion_line = ""

    robot.start()
    cv2.namedWindow("Gretchen Vision")

    try:
        while True:
            ret, img, timestamp = camera.getImage()
            if not ret:
                time.sleep(0.1)
                continue

            results = yolo_model(img, conf=LIVE_CONF_THRESHOLD)[0]
            detections = _parse_detections(results)

            if detections:
                top = max(detections, key=lambda d: d["confidence"])
                category = top["category"]

                if category != last_detected or (time.time() - last_call_time) > COOLDOWN_SECONDS:
                    pil_frame = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    try:
                        last_condition_line, last_suggestion_line = handle_object(top, pil_frame, kb, robot)
                    except wp.ApiCallBudgetExceeded as e:
                        print(f"\n[STOPPED] {e}")
                        break
                    last_detected = category
                    last_call_time = time.time()

            frame = results.plot()
            frame = _draw_overlay(frame, last_condition_line, last_suggestion_line)
            cv2.imshow("Gretchen Vision", frame)
            if cv2.waitKey(1) > 0:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()