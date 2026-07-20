#!/usr/bin/env python3
#
# First Steps in Programming a Humanoid AI Robot
#
# Motor class
# Motor class contains the functions to configure and move the motor
#
#

import os
import datetime
import subprocess

#Need to check if port is available
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import *                    # Uses Dynamixel SDK library

# Control table address
debug = True
ADDR_PRO_TORQUE_ENABLE      = 24               # Control table address is different in Dynamixel model
ADDR_PRO_PRESENT_POSITION   = 37
ADDR_PRO_MOVING_SPEED       = 32
ADDR_PRO_P_GAIN             = 29
# Protocol version
PROTOCOL_VERSION            = 2.0               # See which protocol version is used in the Dynamixel
BAUDRATE                    = 57600             # Dynamixel default baudrate : 57600
# Default setting
#DEVICENAME                  = '/dev/ttyUSB0'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque


################################################################################
#   Motor class
#
#   Input:   (String) device_path - path to device
#            (List) motor_config - a list containing motor configuration dictionaries
#                                  motor0_config = {'Configure_Name_1': (byte_length1, address1, value1), 'Configure_Name_2':
#                                    (byte_length2, address2, value2), #'Configure_Name_3': (byte_length3, address3, value3)}
#                                   For example, {'PGain': (1, 29, 20), 'Speed': (2, 32, 150), 'Torque': (1, 24, 1)}
#            (List) motor_id -  a list of motor ids
#                               For example, [0, 1]
################################################################################

class Motors:
    def __init__(self, device_path, motor_configs, motor_id):
        print('Initializing motors')
        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.portHandler = PortHandler(device_path)

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        self.motor_configs = motor_configs
        self.motor_id = motor_id


    ################################################################################
    #   Opens the port for the motor connection
    #
    #
    #
    #
    ################################################################################

    def openPort(self):
        #global portHandler
        try:
            print ("starting motor controller")
            # Open port
            if self.portHandler.openPort():
                print("Succeeded to open the port")
            else:
                print("Failed to open the port")
                print("Press any key to terminate...")
                getch()
                quit()
        except:
            print("Failed to reconnect to motor...")

        # Set port baudrate
        if self.portHandler.setBaudRate(BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch()
            quit()

    ################################################################################
    #   Configures the motor
    #
    #   Input:   (int) motor id to configure
    #            (dict) containing byte length, address, value
    #            {'Configure_Name_1': (byte_length1, address1, value1), 'Configure_Name_2': (byte_length2, address2, value2),
    #            'Configure_Name_3': (byte_length3, address3, value3)}
    #            {'PGain': (1, 29, 20), 'Speed': (2, 32, 150), 'Torque': (1, 24, 1)}
    #
    ################################################################################

    def setConfig(self, motor_id, motor_config):
        #global portHandler
        global debug

        for i in motor_config:
            byte_length, address, value = motor_config[i]
            #print(byte_length, address, value)
            if byte_length == 1:
                #print("here")
                dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, address, value)
                if dxl_comm_result != COMM_SUCCESS:
                    if debug == True:
                        print("[ID{}] setting {}... failed".format(motor_id, i))
                    quit()
                if dxl_comm_result == COMM_SUCCESS:
                    if debug == True:
                        print("[ID{}] setting {}... success".format(motor_id, i))
            if byte_length == 2:
                dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, address, value)
                if dxl_comm_result != COMM_SUCCESS:
                    if debug == True:
                        print("[ID{}] setting {}... failed".format(motor_id, i))
                    quit()
                if dxl_comm_result == COMM_SUCCESS:
                    if debug == True:
                        print("[ID{}] setting {}... success".format(motor_id, i))

        pass


    ################################################################################
    #   Writes to motor
    #
    #   Input:   (int) motor_id - motor id to write
    #            (int) goal_position_raw- goal raw motor position value
    #
    #
    ################################################################################
    def writeMotor(self, motor_id, goal_position_raw):
        dxl_comm_result = -1000
        while(dxl_comm_result!=COMM_SUCCESS):
            try:
                dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, 30, goal_position_raw)
            except:
                if(debug == True):
                    print("Failed to write packet... retrying...")
                else:
                    pass
            if(dxl_comm_result==COMM_SUCCESS):
                #print("write successful")
                break

    ################################################################################
    #   Reads motor positon values
    #
    #   Input:   (int) motor_id - motor id to read
    #   Returns: (int) motor_position - raw motor position value
    #
    #
    ################################################################################
    def readMotor(self, motor_id):
        dxl_comm_result = -1000
        motor_position = -1
        while(dxl_comm_result!=COMM_SUCCESS):
            try:
                motor_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, motor_id, ADDR_PRO_PRESENT_POSITION)
            except:
                if(debug == True):
                    print("Failed to read packet... retrying...")
                else:
                    pass
        return motor_position

    ################################################################################
    #   Converts a raw motor value to radian value
    #
    #   Input: (int) raw_value - raw motor encoder value
    #   Returns: (float) radian_value - radian value
    #
    ################################################################################
    def convertRaw2Radian(self, raw_value):
        radian_value = float(raw_value-512)/float(512-205)*3.14/2
        return radian_value

    ################################################################################
    #   Converts a radian value to a raw motor value
    #
    #   Input: (float) radian_value - radian value
    #   Returns: (int) raw_value - raw motor encoder value
    #
    ################################################################################
    def convertRadian2Raw(self, radian_value):
        raw_value = int(radian_value*2/3.14*float(512-205)+512)
        return raw_value

    ################################################################################
    #   Closes the port for the motor connection
    #
    #
    #
    #
    ################################################################################
    def closeMotor(self):
        try:
            self.portHandler.closePort()
            print ("Disabled motors and closed ports ")
        except:
            pass

def main():
    motor_configs = []
    motor0_config, motor1_config = {}, {}
    motor0_config["PGain"] = (1, ADDR_PRO_P_GAIN, 20)
    motor0_config["Speed"] = (2, ADDR_PRO_MOVING_SPEED, 500)
    motor0_config["Torque"] = (1, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)

    motor1_config["PGain"] = (1, ADDR_PRO_P_GAIN, 50)
    motor1_config["Speed"] = (2, ADDR_PRO_MOVING_SPEED, 500)
    motor1_config["Torque"] = (1, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)

    motor_configs.append(motor0_config)
    motor_configs.append(motor1_config)

    motor_id = [0, 1]
    device="/dev/grt_motor"
    motors = Motors(device, motor_configs, motor_id)

    motors.openPort()

    #print("Setting configurations...")
    for i, motor_config in enumerate(motor_configs):
        motors.setConfig(i, motor_config)
    #print("Reading motor positions...")
    motor_position_raw, motor_position_rad = {}, {}
    for i in motor_id:
        motor_position_raw[i] = motors.readMotor(i)
        motor_position_rad[i] = motors.convertRaw2Radian(motor_position_raw[i])
    #print("Writing motor positions...")
    #print(motors.convertRadian2Raw(1.5))
    motor_goal_rad = {0:.0, 1:.0}
    motor_goal_raw = {}
    for i in motor_id:
        motor_goal_raw[i] = motors.convertRadian2Raw(motor_goal_rad[i])
        motors.writeMotor(i, motor_goal_raw[i])

    #print(motor_position_raw)
    #print(motor_position_rad)
    #readMotor()
    motors.closeMotor()
    pass


if __name__=="__main__":
    main()
