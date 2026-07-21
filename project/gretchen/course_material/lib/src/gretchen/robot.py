#!/usr/bin/env python3
#
# First Steps in Programming a Humanoid AI Robot
#
# Robot class
# Robot class contains functions to move the robot and recieve inputs from
# the robot's camera.
#

import cv2
import numpy as np
import math
import time

from gretchen.motors import Motors
from gretchen.camera import Camera
from datetime import datetime



ADDR_PRO_P_GAIN             = 29
ADDR_PRO_MOVING_SPEED       = 32
ADDR_PRO_TORQUE_ENABLE      = 24
TORQUE_ENABLE               = 1                 # Value for enabling the torque

################################################################################
#   Robot class
#
#
#
#
################################################################################
class Robot:
    def __init__(self, motor_device_path = 'COM3', camera_device_path = 0):
        print('Initializing robot')

        #Path to motor/camera device
        #Ubuntu/Linux Path
        self.motor_device_path= motor_device_path
        self.camera_device_path = camera_device_path

        #Windows Path
        #motor_device_path= 'COM4'
        #camera_device_path = 0

        #Mac Path
        #motor_device_path= '/dev/tty.usbserial-FT5WJ4JS'
        #camera_device_path = 0 #'/dev/cu.usbserial-FT5WJ4JS'

        #List of motor ids
        self.motor_id = [0, 1]

        #Configurations for the motors
        #self.motor_configs is a list containing motor configuration dictionaries
        #motor0_config = {'Configure_Name_1': (byte_length1, address1, value1), 'Configure_Name_2': (byte_length2, address2, value2), 'Configure_Name_3': (byte_length3, address3, value3)}
        #For example, {'PGain': (1, 29, 20), 'Speed': (2, 32, 150), 'Torque': (1, 24, 1)}
        motor0_config, motor1_config = {}, {}
        motor0_config["PGain"] = (1, ADDR_PRO_P_GAIN, 20)
        #motor0_config["Speed"] = (2, ADDR_PRO_MOVING_SPEED, 500)
        motor0_config["Speed"] = (2, ADDR_PRO_MOVING_SPEED, 150)
        motor0_config["Torque"] = (1, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)
        motor1_config["PGain"] = (1, ADDR_PRO_P_GAIN, 50)
        #motor1_config["Speed"] = (2, ADDR_PRO_MOVING_SPEED, 500)
        motor1_config["Speed"] = (2, ADDR_PRO_MOVING_SPEED, 150)
        motor1_config["Torque"] = (1, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)

        #Contains motor configurations in a list
        self.motor_configs = []
        self.motor_configs.append(motor0_config)
        self.motor_configs.append(motor1_config)

        #Create Motor class instance
        self.motors = Motors(self.motor_device_path, self.motor_configs, self.motor_id)

        #Create Camera class instance
        self.camera = Camera(self.camera_device_path)

        #Contains raw current motor position values, raw goal motor position values
        self.motor_position_raw, self.motor_position_rad = {}, {}
        self.motor_goal_raw, self.motor_goal_rad= {}, {}

        #moving variable
        self.moving = False

        #Goal position for pan/tilt
        self.goal_theta0 = None
        self.goal_theta1 = None

        #Current position for pan/tilt
        self.cur_theta0 = None
        self.cur_theta1 = None

        #Gretchen robot structure
        self.d0 = 0.15
        self.d1 = 0
        self.d2 = 0.5

        self.dt = datetime.now()
        self.ts = datetime.timestamp(self.dt)

    ################################################################################
    #   Starts robot motor - Opens motor device port and sets the motor configurations
    #   Starts robot camera
    #
    #
    #
    #
    ################################################################################
    def start(self):
        self.start_motors()
        self.start_camera()


    ################################################################################
    #   Starts the robot camera
    #
    #
    #
    #
    #
    ################################################################################
    def start_camera(self):
        #Start camera
        self.camera.start()


    ################################################################################
    #   Opens motor device port and sets the motor configurations
    #
    #
    #
    #
    #
    ################################################################################
    def start_motors(self):
        #Start motor
        self.motors.openPort()
        print("Setting motor configurations...")
        for i, motor_config in enumerate(self.motor_configs):
            self.motors.setConfig(i, motor_config)

        self.goal_theta0, self.goal_theta1 = self.getPosition()
        self.cur_theta0, self.cur_theta1 = self.getPosition()

    ################################################################################
    #   Disconnects from the motors
    #
    #
    #
    #
    #
    ################################################################################
    def disconnect(self):
        self.motors.closeMotor()

    ################################################################################
    #   Gets current position of the robot
    #
    #   Returns: (List) [theta]
    #            (int) goal raw motor position value
    #
    #
    ################################################################################
    def getPosition(self):
        #return [self.cur_pan_angle, self.cur_tilt_angle]
        #print("Reading motor positions...")
        motor_position_rad_list = []
        for i in self.motor_id:
            self.motor_position_raw[i] = self.motors.readMotor(i)
            self.motor_position_rad[i] = self.motors.convertRaw2Radian(self.motor_position_raw[i])
            motor_position_rad_list.append(self.motor_position_rad[i])
        return motor_position_rad_list

    ################################################################################
    #   Moves to the motors
    #
    #   Input:   (float) pan_rad - pan angle to move in radians
    #            (float) tilt_rad - tilt angle to move in radians
    #
    #   Returns: True to indicate moving finished
    #
    ################################################################################
    def move(self, pan_rad, tilt_rad):
        #print("Writing motor positions...")
        #print(motors.convertRadian2Raw(1.5))
        self.motor_goal_rad = {0:pan_rad, 1:tilt_rad}
        #motor_goal_raw = {}
        for i in self.motor_id:
            self.motor_goal_raw[i] = self.motors.convertRadian2Raw(self.motor_goal_rad[i])
            self.motors.writeMotor(i, self.motor_goal_raw[i])
        return True

    def reachGoal(self, threshold):
        goal_reached = False
        #print("self.goal_theta0, self.goal_theta1",self.goal_theta0, self.goal_theta1 )
        self.cur_theta0, self.cur_theta1 = self.getPosition()
        distance = math.sqrt((self.cur_theta0-self.goal_theta0)**2+(self.cur_theta1-self.goal_theta1)**2)
        #print(distance)
        if (distance < threshold):
            goal_reached = True
        return goal_reached

    ################################################################################
    #   Moves the robot to point at the goal x, y, z coordinates
    #   We use inverse kinematics to compute the pan/tilt angle
    #   In order to compute the inverse kinematics, we need to compute the
    #   distance between the robot base and the goal x, y, z coordinates
    #
    #   Input:   (int) goal x coordinate in respect to the robot
    #            (int) goal y coordinate in respect to the robot
    #            (int) goal z coordinate in respect to the robot
    #
    #   Returns: (float) theta0 - goal pan radian
    #            (float) theta1 - goal tilt radian
    #
    ################################################################################
    def lookatpoint(self, x, y, z):
        #Gets current pan/tilt position
        theta0, theta1 = self.getPosition()

        #Robot homography
        robot_H0, robot_H1, robot_H2 = self.forwardKinematics(theta0, theta1, self.d0, self.d1, self.d2)

        #H0_R, H1_R, H2_R
        #robot_H0, robot_H1, robot_H2
        #Compute camera homography
        #Center values of an image.
        #We want to find the distance between camera and the center of the image of the camera
        u =319.5
        v =239.5

        #Center of the image's x, y, z coordinates in respect to the camera
        camera_x, camera_y, camera_z = self.convert2d_3d(u,v)
        #Camera homography
        camera_H = np.array([[1, 0, 0, camera_z],
                        [0, 1, 0, -camera_y],
                        [0, 0, 1, camera_x],
                        [0, 0, 0, 1]])

        #Need to find camera distance for this to work
        goal_robot_camera_H = np.array([[1, 0, 0, x],
                        [0, 1, 0, y],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]])

        #Find the robot base position
        robot_base_H = robot_H0 @ robot_H1

        #Compute the camera distance
        camera_distance_matrix = goal_robot_camera_H - robot_base_H

        #Compute euclidiean camera distance
        camera_distance = math.sqrt(camera_distance_matrix[0,3]**2 + camera_distance_matrix[1,3]**2+ camera_distance_matrix[2,3]**2)

        #Camera distance is used to compute the inverse kinematics
        theta0, theta1 = self.inverseKinematics(x, y, z, self.d0, self.d1, camera_distance)

        #Update goal position
        self.goal_theta0 = theta0
        self.goal_theta1 = theta1

        #Recieves true if moved
        moved = self.move(theta0, theta1)

        return theta0, theta1, moved

    ################################################################################
    #   Converts 2D u, v camera coordinates to 3D x, y, z coordinates in respect to the camera
    #
    #   Input:   (int) u - coordinate in a camera image
    #            (int) v - coordinate in a camera image
    #
    #   Returns: (int) x - coordinate in respect to the camera
    #            (int) y - coordinate in respect to the camera
    #            (int) z - coordinate in respect to the camera
    #
    ################################################################################
    def convert2d_3d(self, u, v):
        #camera.fx = 570.3422241210938*2
        #camera.fy = 570.3422241210938*2
        #camera.cx = 319.5
        #camera.cy = 239.5

        x =  ( u  - self.camera.cx )/ self.camera.fx
        y =  ( v  - self.camera.cy )/ self.camera.fy
        #We assume the distance z is 0.50
        z = 0.50

        return (x, y, z)

    ################################################################################
    #   Converts 3D x, y, z coordinates in respect to the camera to
    #            3D x, y, z coordinates in respect to the robot
    #
    #   Input:   (int) x - coordinate in respect to the camera
    #            (int) y - coordinate in respect to the camera
    #            (int) z - coordinate in respect to the camera
    #
    #   Returns: (int) x - coordinate in respect to the robot
    #            (int) y - coordinate in respect to the robot
    #            (int) z - coordinate in respect to the robot
    #
    ################################################################################
    def convert3d_3d(self, x, y, z, robot_H):
        camera_H = np.array([[1, 0, 0, z],
                        [0, 1, 0, -y],
                        [0, 0, 1, x],
                        [0, 0, 0, 1]])
        #Projects in respect to the roboto homography
        H = robot_H @ camera_H
        return H[0, 3], H[1, 3], H[2, 3]

    ################################################################################
    #   forwardKinematics helper function that creates rotation portion of the matrix
    #
    #   Input:   (int) theta - the angle amount to rotate
    #            (list) [x, y, z] - axis indicating which to rotate
    #            i.e. [1, 0, 0] means rotating on x
    #
    #   Returns: (np.array) rotation matrix
    #
    ################################################################################
    def forwardKinematicsRot(self, theta, axis):
        #axis = [x, y, z]
        R_z = np.array([[math.cos(theta), -math.sin(theta), 0, 0],
                        [math.sin(theta), math.cos(theta), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        R_y = np.array([[math.cos(theta), 0, math.sin(theta), 0],
                        [0, 1, 0, 0],
                        [-math.sin(theta), 0, math.cos(theta), 0],
                        [0, 0, 0, 1]])
        R_x = np.array([[1, 0, 0, 0],
                        [0, math.cos(theta), -math.sin(theta), 0],
                        [0, math.sin(theta), math.cos(theta), 0],
                        [0, 0, 0, 1]])
        R = R_x*axis[0] + R_y*axis[1] + R_z*axis[2]
        return R

    ################################################################################
    #   forwardKinematics helper function that creates translation portion of the matrix
    #
    #   Input:   (int) delta - the amount to translate
    #            (list) [x, y, z] - axis indicating which to translate
    #            i.e. [1, 0, 0] means translating on x
    #
    #   Returns: (np.array) translation matrix
    #
    ################################################################################
    def forwardKinematicsTrans(self, delta, axis):
        #axis = [x, y, z]
        T_z = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, delta],
                        [0, 0, 0, 0]])
        T_y = np.array([[0, 0, 0, 0],
                        [0, 0, 0, delta],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
        T_x = np.array([[0, 0, 0, delta],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        T = T_x*axis[0] + T_y*axis[1] + T_z*axis[2]

        return T

    ################################################################################
    #   forwardKinematics for the robot
    #
    #   Input:   (float) theta0 - pan angle
    #            (float) theta1 - tilt angle
    #            (float) delta0 - link length, 0.15
    #            (float) delta1 - link length, 0
    #            (float) delta2 - link length, 0.5
    #
    #   Returns: (np.array) - robot link0 homography
    #            (np.array) - robot link1 homography
    #            (np.array) - robot link2 homography
    #
    ################################################################################
    def forwardKinematics(self, theta0, theta1, delta0, delta1, delta2):
        #Compute H0_1
        #Dynamic rotation theta0 around Z axis
        link0_0 = self.forwardKinematicsRot(theta0, [0,0,1]) + self.forwardKinematicsTrans(delta0, [0,0,1])
        #Static rotation 90 degree around X axis
        link0_1 = self.forwardKinematicsRot(90*math.pi/180, [1,0,0])
        H0_1 = link0_0 @ link0_1

        #Compute H1_2
        #Dynamic rotation theta1 around Z axis
        link1 = self.forwardKinematicsRot(theta1, [0, 0, 1]) + self.forwardKinematicsTrans(delta1, [1,0,0])
        H1_2 = link1

        #Static translation delta2 on X axis
        link2 = self.forwardKinematicsRot(0, [0, 0, 1]) +self.forwardKinematicsTrans(delta2, [1,0,0])
        H2_3 = link2

        return H0_1, H1_2, H2_3


    ################################################################################
    #   forwardKinematics for the robot
    #
    #   Input:   (int) x - goal x coordinate in respect to the robot
    #            (int) y - goal y coordinate in respect to the robot
    #            (int) z - goal z coordinate in respect to the robot
    #            (int) d0 - length of first link
    #            (int) d1 - length of second link
    #            (int) d2 - length of third link
    #
    #   Returns: (int) theta0 - goal pan angle
    #            (int) theta1 - goal tilt angle
    #
    ################################################################################
    def inverseKinematics(self, x, y, z, d0, d1, d2):
        #1. Compute theta1
        #General Case - tilt under 90 degrees
        theta1 = math.asin((z-d0)/(d2+1e-10))

        #Special case - tilt over 90 degrees
        theta1_1 = math.acos((z-d0)/(d2+1e-10)) #Postiive case
        theta1_2 = math.acos((d0-z)/(d2+1e-10)) #Negative case

        #Special Case - tilt over 90 degrees
        if x <= d1:
            #Depends on the z value
            if z > 0:   #Postiive case
                theta1 = ((theta1_1)+90*math.pi/180)
            else:       #Negative case
                theta1 = -1*theta1_2-90*math.pi/180

        #2. Compute theta0
        theta0_x = math.acos((x/(math.cos(theta1)+1e-10)-d1)/d2)
        theta0_y = math.asin(y/(1e-10+d1+d2*(math.cos(theta1))))
        theta0 = theta0_x * (theta0_y/abs(theta0_y+1e-10))
        return theta0, theta1

    ################################################################################
    #   Return pan angle
    #
    #   Returns: (int) self.cur_theta0 - current pan angle
    #
    ################################################################################
    def getPanAngle(self):
        self.cur_theta0, self.cur_theta1 = self.getPosition()
        return self.cur_theta0

    ################################################################################
    #   Return tilt angle
    #
    #   Returns: (int) self.cur_theta1 - current tilte angle
    #
    ################################################################################
    def getTiltAngle(self):
        self.cur_theta0, self.cur_theta1 = self.getPosition()
        return self.cur_theta1

    ################################################################################
    #   Moves the robot to point to the center
    #
    #
    ################################################################################
    def center(self):
        self.move(0, 0)

    ################################################################################
    #   Moves the robot down based on relative position
    #
    #   Input:   (float) delta - amount to move
    #
    ################################################################################
    def down(self, delta=0.1, resend=False):
        self.cur_theta0, self.cur_theta1 = self.getPosition()
        goal_theta1 = self.cur_theta1 - delta
        self.move(self.cur_theta0, goal_theta1)

    ################################################################################
    #   Moves the robot up based on relative position
    #
    #   Input:   (float) delta - amount to move
    #
    ################################################################################
    def up(self, delta=0.1, resend=False):
        self.cur_theta0, self.cur_theta1 = self.getPosition()
        goal_theta1 = self.cur_theta1 + delta
        self.move(self.cur_theta0, goal_theta1)

    ################################################################################
    #   Moves the robot left based on relative position
    #
    #   Input:   (float) delta - amount to move
    #
    ################################################################################
    def left(self, delta=0.1, resend=False):
        self.cur_theta0, self.cur_theta1 = self.getPosition()
        goal_theta0 = self.cur_theta0 - delta
        self.move(goal_theta0, self.cur_theta1)

    ################################################################################
    #   Moves the robot right based on relative position
    #
    #   Input:   (float) delta - amount to move
    #
    ################################################################################
    def right(self, delta=0.1, resend=False):
        self.cur_theta0, self.cur_theta1 = self.getPosition()
        goal_theta0 = self.cur_theta0 + delta
        self.move(goal_theta0, self.cur_theta1)



def main():
    robot = Robot()
    robot.start()
    current_rad = robot.getPosition()
    #print(current_rad)
    robot.move(0.0, 0.0)
    current_rad = robot.getPosition()
    #print(current_rad)
    #print(robot.motor_configs)

    #print(robot.getPanAngle())
    #print(robot.getTiltAngle())
    robot.center()
if __name__=="__main__":
    main()
