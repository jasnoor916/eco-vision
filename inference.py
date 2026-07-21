import argparse
import os
from pathlib import Path

import cv2
from dotenv import load_dotenv


# Load defaults from the .env beside this file. Command-line arguments override
# these values when provided.
load_dotenv(Path(__file__).resolve().parent / ".env")


def camera_source(value):
    """Convert a numeric camera index to int; keep device paths as strings."""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run live YOLO inference with a laptop camera or the robot."
    )
    parser.add_argument(
        "--mode",
        choices=("laptop", "robot"),
        default="robot",
        help="Use OpenCV directly (laptop) or initialize the robot and motors (robot).",
    )
    parser.add_argument(
        "--camera",
        type=camera_source,
        default=None,
        help="Camera index or device path (default: 0 for laptop; ROBOT_CAMERA for robot).",
    )
    parser.add_argument(
        "--motor-port",
        default=os.getenv("ROBOT_MOTOR_PORT", "/dev/tty.usbserial-FT94ELHH"),
        help="Robot motor serial port (used only in robot mode).",
    )
    return parser.parse_args()


def open_camera(args):
    if args.mode == "laptop":
        source = 0 if args.camera is None else args.camera
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open laptop camera: {source}")

        def read_frame():
            return capture.read()

        return read_frame, capture.release, source

    # Importing Robot also imports the motor SDK, so keep it out of laptop mode.
    from gretchen.robot import Robot

    source = (
        camera_source(os.getenv("ROBOT_CAMERA", "0"))
        if args.camera is None
        else args.camera
    )
    robot = Robot(args.motor_port, source)
    robot.start()

    def read_frame():
        ret, frame, _timestamp = robot.camera.getImage()
        return ret, frame

    def close_robot():
        if robot.camera.vc is not None:
            robot.camera.vc.release()
        robot.disconnect()

    return read_frame, close_robot, source


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO("yolo26s.pt")
    read_frame, close_camera, source = open_camera(args)
    print(f"Running in {args.mode} mode with camera {source!r}. Press any key to exit.")

    try:
        cv2.namedWindow("Frame")
        while True:
            ret, img = read_frame()
            if not ret or img is None:
                raise RuntimeError(f"Failed to read a frame from camera: {source}")

            results = model(img)
            cv2.imshow("Frame", results[0].plot())

            if cv2.waitKey(1) > 0:
                break
    finally:
        close_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
