import cv2
import numpy as np
from ultralytics import YOLO
from gretchen.robot import Robot

model = YOLO("yolo26s.pt")
robot = Robot('COM3', 0)
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