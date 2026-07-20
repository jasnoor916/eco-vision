#!/usr/bin/env python3
#
# First Steps in Programming a Humanoid AI Robot
#
# Camera class
#
#
#

import cv2
from datetime import datetime
################################################################################
#   Camera class
#
#   Input:   (String) path to device
#
#
################################################################################

class Camera:
    def __init__(self, path):

        #Camera parameters
        self.fx = 570.3422241210938*2
        self.fy = 570.3422241210938*2
        self.cx = 319.5
        self.cy = 239.5

        #VideoCapture
        self.vc = None

        #timestamp
        self.dt = datetime.now()
        self.ts = datetime.timestamp(self.dt)

        #Path to camera device
        self.path = path

    ################################################################################
    #   Starts camera inputs
    #
    #
    #
    #
    ################################################################################
    def start(self):
        self.vc = cv2.VideoCapture(self.path)

    ################################################################################
    #   Gets image frame
    #
    #   Returns:   (Boolean) ret - True, if read properly
    #              (np.array) frame - image frame
    #              (float) self.ts - timestamp
    #
    #
    ################################################################################
    def getImage(self):
        ret, frame = self.vc.read()
        self.dt = datetime.now()
        self.ts = datetime.timestamp(self.dt)

        #Resize
        frame = cv2.resize(frame, (640, 480))
        return ret, frame, self.ts

def main():
    print("Hello World")
    cv2.namedWindow("Camera")
    camera = Camera('/dev/grt_cam')
    vc = camera.start()
    while True:
        ret, frame, ts = camera.getImage()
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1)
        if key > 0:
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
