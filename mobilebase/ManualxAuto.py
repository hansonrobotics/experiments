import time
import pyrealsense2 as rs
import numpy as np
import cv2
import pygame
from pyvesc import VESC
from enum import Enum
import math
import csv

# ================== CONFIGURATION ==================
# --- Serial Ports ---
LEFT_MOTOR_SERIAL_PORT = "/dev/left_motor"
RIGHT_MOTOR_SERIAL_PORT = "/dev/right_motor"

# --- Motor Control Settings ---
MAX_RPM = 1100
FORWARD_RPM = 900
TURN_RPM = 900
BRAKE_RPM = 0
REVERSE_RPM = -900

# --- Joystick Settings ---
JOYSTICK_THROTTLE_AXIS = 1
JOYSTICK_STEER_AXIS = 0
DEAD_MAN_SWITCH_BUTTON = 5  # Trigger
MODE_SWITCH_BUTTON = 0      # The 'X' button on a PS3 controller
JOYSTICK_DEADZONE = 0.45
SMOOTHING_FACTOR = 0.065
DECELERATION_FACTOR = 0.05

# --- Autonomous Behavior ---
REACTION_DISTANCE = 1.0
EMERGENCY_DISTANCE = 0.3
BRAKE_DURATION = 0.4
REVERSE_DURATION = 0.6
# ===================================================

class RobotMode(Enum):
    MANUAL = 1
    AUTONOMOUS = 2

class AutonomousState(Enum):
    DRIVING_FORWARD = 1
    STOPPING = 2
    SCANNING_TURN = 3
    REVERSING = 4

class VescMotor: #A class to represent and control a single VESC motor
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
    # --- Initialization ---
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("❌ ERROR: No joystick detected.")
        return
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"✅ Joystick '{joystick.get_name()}' initialized.")

    print("Starting RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
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

    # --- Initialize Motors ---
    print("Attempting to connect to motors...")
    left_motor = VescMotor(LEFT_MOTOR_SERIAL_PORT)
    right_motor = VescMotor(RIGHT_MOTOR_SERIAL_PORT)
    if not left_motor.driver or not right_motor.driver:
        pipeline.stop()
        cv2.destroyAllWindows()
        return

    # --- Initialize State Machines & Data Logger ---
    robot_mode = RobotMode.MANUAL
    auton_state = AutonomousState.DRIVING_FORWARD
    state_change_time = time.time()
    current_throttle = 0.0
    current_steer = 0.0
    
    data_log_file = open('training_data.csv', 'w', newline='')
    csv_writer = csv.writer(data_log_file)
    csv_writer.writerow(['timestamp', 'depth_center', 'joystick_throttle', 'joystick_steer'])

    print("\n🚀 System live. Manual control enabled. Hold trigger to drive.")
    print("   Press the 'X' button to switch modes.")

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # Check for the mode switch button press
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == MODE_SWITCH_BUTTON:
                        if robot_mode == RobotMode.MANUAL:
                            robot_mode = RobotMode.AUTONOMOUS
                            auton_state = AutonomousState.DRIVING_FORWARD # Reset auton state
                            print("\nSwitched to AUTONOMOUS mode.")
                        else:
                            robot_mode = RobotMode.MANUAL
                            print("\nSwitched to MANUAL mode.")

            # --- Get Camera Data ---
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            height, width, _ = color_image.shape
            
            center_zone = depth_image[:, width//3:2*width//3]
            non_zero_center = center_zone[center_zone > 0]
            closest_center = non_zero_center.min() * depth_frame.get_units() if non_zero_center.size > 0 else float('inf')

            # --- Main Control Logic ---
            if robot_mode == RobotMode.AUTONOMOUS:
                if auton_state == AutonomousState.DRIVING_FORWARD:
                    left_motor.set_rpm(FORWARD_RPM)
                    right_motor.set_rpm(FORWARD_RPM)
                    if 0 < closest_center < EMERGENCY_DISTANCE:
                        auton_state = AutonomousState.REVERSING
                        state_change_time = time.time()
                    elif 0 < closest_center < REACTION_DISTANCE:
                        auton_state = AutonomousState.STOPPING
                        state_change_time = time.time()
                elif auton_state == AutonomousState.REVERSING:
                    left_motor.set_rpm(REVERSE_RPM)
                    right_motor.set_rpm(REVERSE_RPM)
                    if time.time() - state_change_time > REVERSE_DURATION:
                        auton_state = AutonomousState.STOPPING
                        state_change_time = time.time()
                elif auton_state == AutonomousState.STOPPING:
                    left_motor.set_rpm(BRAKE_RPM)
                    right_motor.set_rpm(BRAKE_RPM)
                    if time.time() - state_change_time > BRAKE_DURATION:
                        auton_state = AutonomousState.SCANNING_TURN
                elif auton_state == AutonomousState.SCANNING_TURN:
                    if closest_center > REACTION_DISTANCE:
                        auton_state = AutonomousState.DRIVING_FORWARD
                    else:
                        left_motor.set_rpm(TURN_RPM)
                        right_motor.set_rpm(-TURN_RPM)

            elif robot_mode == RobotMode.MANUAL:
                is_enabled = joystick.get_button(DEAD_MAN_SWITCH_BUTTON)
                target_throttle = 0.0
                target_steer = 0.0
                smoothing_to_use = DECELERATION_FACTOR

                if is_enabled:
                    target_throttle = -joystick.get_axis(JOYSTICK_THROTTLE_AXIS)
                    target_steer = joystick.get_axis(JOYSTICK_STEER_AXIS)
                    smoothing_to_use = SMOOTHING_FACTOR
                    csv_writer.writerow([time.time(), closest_center, target_throttle, target_steer])

                magnitude = math.hypot(target_throttle, target_steer)
                if magnitude < JOYSTICK_DEADZONE:
                    target_throttle = 0.0
                    target_steer = 0.0
                
                current_throttle += (target_throttle - current_throttle) * smoothing_to_use
                current_steer += (target_steer - current_steer) * smoothing_to_use

                if abs(current_throttle) < 0.01: current_throttle = 0.0
                if abs(current_steer) < 0.01: current_steer = 0.0

                left_raw = current_throttle + current_steer
                right_raw = current_throttle - current_steer

                max_raw = max(abs(left_raw), abs(right_raw))
                if max_raw > 1.0:
                    left_speed = left_raw / max_raw
                    right_speed = right_raw / max_raw
                else:
                    left_speed = left_raw
                    right_speed = right_raw

                left_motor.set_rpm(left_speed * MAX_RPM)
                right_motor.set_rpm(right_speed * MAX_RPM)

            # --- Visualization ---
            cv2.line(color_image, (width//3, 0), (width//3, height), (0, 255, 255), 2)
            cv2.line(color_image, (2*width//3, 0), (2*width//3, height), (0, 255, 255), 2)
            cv2.imshow("RealSense Viewer", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                running = False

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user.")
    finally:
        print("\nStopping motors and shutting down.")
        if 'left_motor' in locals() and left_motor.driver: left_motor.set_rpm(0)
        if 'right_motor' in locals() and right_motor.driver: right_motor.set_rpm(0)
        pipeline.stop()
        cv2.destroyAllWindows()
        data_log_file.close()
        print("Data logging complete. 'training_data.csv' saved.")
        pygame.quit()

if __name__ == '__main__':
    main()