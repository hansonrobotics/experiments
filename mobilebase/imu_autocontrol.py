import time
import pyrealsense2 as rs
import numpy as np
import cv2
from pyvesc import VESC
from enum import Enum

# ================== CONFIGURATION ==================
# --- Serial Ports ---
LEFT_MOTOR_SERIAL_PORT = "/dev/left_motor"
RIGHT_MOTOR_SERIAL_PORT = "/dev/right_motor"

# --- Motor Control Settings ---
FORWARD_RPM = 900
TURN_RPM = 800
BRAKE_RPM = 0
REVERSE_RPM = -900

# --- Robot Behavior ---
REACTION_DISTANCE = 1.0
EMERGENCY_DISTANCE = 0.3
BRAKE_DURATION = 0.4
REVERSE_DURATION = 0.6

# --- IMU Heading Correction ---
YAW_CORRECTION_GAIN = 100.0
GYRO_DEADZONE = 0.02
# ===================================================

class RobotState(Enum):
    DRIVING_FORWARD = 1
    STOPPING = 2
    SCANNING_TURN = 3
    REVERSING = 4

class VescMotor:
    """A class to represent and control a single VESC motor."""
    def __init__(self, serial_port):
        self.serial_port = serial_port
        self.driver = None
        try:
            self.driver = VESC(serial_port=self.serial_port)
            print(f"✅ VESC connected on {self.serial_port}")
        except Exception as e:
            print(f"❌ ERROR connecting to VESC on {self.serial_port}: {e}")

    def set_rpm(self, rpm):
        if self.driver:
            self.driver.set_rpm(int(rpm))

def main():
    # --- Phase 1: Initialize Vision System ---
    print("Starting RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    # Use lower resolution and framerate to reduce power consumption
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("✅ RealSense camera started.")
    except Exception as e:
        print(f"❌ ERROR starting camera: {e}")
        return

    # --- Show the first frame BEFORE connecting to motors ---
    print("Displaying viewer... Press 'q' in the window to exit.")
    try:
        frames = pipeline.wait_for_frames(5000)
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("RealSense Viewer", color_image)
        cv2.waitKey(1)
    except Exception as e:
        print(f"❌ ERROR displaying first frame: {e}")
        pipeline.stop()
        return

    # --- Phase 2: Initialize Motors ---
    print("Attempting to connect to motors...")
    left_motor = VescMotor(LEFT_MOTOR_SERIAL_PORT)
    right_motor = VescMotor(RIGHT_MOTOR_SERIAL_PORT)
    if not left_motor.driver or not right_motor.driver:
        pipeline.stop()
        cv2.destroyAllWindows()
        return

    # --- Initialize State Machine ---
    robot_state = RobotState.DRIVING_FORWARD
    state_change_time = time.time()
    print("\n🚀 Autonomous mode is live.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            
            # --- Get Sensor Data ---
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            height, width, _ = color_image.shape

            # --- Three-Zone Bumper Check (always active) ---
            center_zone = depth_image[:, width//3:2*width//3]
            non_zero_center = center_zone[center_zone > 0]
            closest_center = non_zero_center.min() * depth_frame.get_units() if non_zero_center.size > 0 else float('inf')

            # --- State Machine Logic ---
            if robot_state == RobotState.DRIVING_FORWARD:
                # --- IMU-based heading correction ---
                yaw_rate = 0.0
                if gyro_frame:
                    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                    yaw_rate = gyro_data.z
                
                if abs(yaw_rate) < GYRO_DEADZONE:
                    yaw_rate = 0.0

                correction_rpm = yaw_rate * YAW_CORRECTION_GAIN
                
                left_rpm = FORWARD_RPM - correction_rpm
                right_rpm = FORWARD_RPM + correction_rpm

                print(f"State: DRIVING | Yaw Rate: {yaw_rate:.2f} | Correction: {int(correction_rpm)} RPM", end='\r')
                left_motor.set_rpm(left_rpm)
                right_motor.set_rpm(right_rpm)

                if 0 < closest_center < EMERGENCY_DISTANCE:
                    robot_state = RobotState.REVERSING
                    state_change_time = time.time()
                elif 0 < closest_center < REACTION_DISTANCE:
                    robot_state = RobotState.STOPPING
                    state_change_time = time.time()

            elif robot_state == RobotState.REVERSING:
                print("State: REVERSING      ", end='\r')
                left_motor.set_rpm(REVERSE_RPM)
                right_motor.set_rpm(REVERSE_RPM)
                if time.time() - state_change_time > REVERSE_DURATION:
                    robot_state = RobotState.STOPPING
                    state_change_time = time.time()

            elif robot_state == RobotState.STOPPING:
                print("State: STOPPING       ", end='\r')
                left_motor.set_rpm(BRAKE_RPM)
                right_motor.set_rpm(BRAKE_RPM)
                if time.time() - state_change_time > BRAKE_DURATION:
                    robot_state = RobotState.SCANNING_TURN

            elif robot_state == RobotState.SCANNING_TURN:
                print("State: SCANNING_TURN  ", end='\r')
                if closest_center > REACTION_DISTANCE:
                    print("\nPath is clear. Resuming forward drive.")
                    robot_state = RobotState.DRIVING_FORWARD
                else:
                    left_motor.set_rpm(TURN_RPM)
                    right_motor.set_rpm(-TURN_RPM)
            
            # --- Visualization ---
            cv2.line(color_image, (width//3, 0), (width//3, height), (0, 255, 255), 2)
            cv2.line(color_image, (2*width//3, 0), (2*width//3, height), (0, 255, 255), 2)
            cv2.imshow("RealSense Viewer", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user.")
    finally:
        print("\nStopping motors and shutting down.")
        if 'left_motor' in locals() and left_motor.driver: left_motor.set_rpm(0)
        if 'right_motor' in locals() and right_motor.driver: right_motor.set_rpm(0)
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()