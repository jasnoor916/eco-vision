import os
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
from gretchen.robot import Robot

# Load robot settings from the .env at the repo root (this file lives at the
# root, so .env sits beside it). See .env for all options and per-OS values.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _camera(value):
    """A plain number is a camera index (int); anything else is a device path."""
    return int(value) if value.isdigit() else value


ROBOT_MOTOR_PORT = os.getenv("ROBOT_MOTOR_PORT", "/dev/tty.usbserial-FT94ELHH")
ROBOT_CAMERA = _camera(os.getenv("ROBOT_CAMERA", "0"))

model = YOLO("yolo26s.pt")
robot = Robot(ROBOT_MOTOR_PORT, ROBOT_CAMERA)
camera = robot.camera

def main():
    robot.start()

    cv2.namedWindow("Frame")

    while True:
        # Get image from camera
        ret, img, timestamp = camera.getImage()

        # Run YOLO
        results = model(img)

        # Draw detections
        annotated_img = results[0].plot()

        # Show annotated image
        cv2.imshow("Frame", annotated_img)

        # Exit on any key
        if cv2.waitKey(1) > 0:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()