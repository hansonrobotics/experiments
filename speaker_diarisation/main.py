#!/usr/bin/env python3

import rospy, os, sys

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) #scripts directory
    os.environ['DEEPFACE_HOME'] = project_root
    os.environ['TORCH_HOME'] = project_root
    os.environ['PROJECT_ROOT'] = project_root
except Exception as e:
    print(f"CRITICAL: Could not set environment variables. Error: {e}", file=sys.stderr)
    sys.exit(1)

from realsense.fusion_engine import FusionEngine

if __name__ == '__main__':
    print("--> Main script started.")
    node = None
    try:
        rospy.init_node('fusion_engine_node', anonymous=True)
        print("--> ROS node initialized.")

        node = FusionEngine()
        print("--> FusionEngine initialized. Starting run loop... Press Ctrl+C to exit.")

        node.run()

    except rospy.ROSInterruptException:
        print("\n--> ROSInterruptException (Ctrl+C). Shutting down...")
    except Exception as e:

        if rospy.is_shutdown():
            print(f"FATAL ERROR during shutdown: {e}", file=sys.stderr)
        else:
            rospy.logfatal(f"Unhandled exception in main execution: {e}", exc_info=True)
            print(f"FATAL ERROR: {e}", file=sys.stderr)
    finally:
        print("--> Reached finally block in main.")
        if node:
            print("--> Calling node stop() for cleanup.")
            node.stop()
        else:
             print("--> Node object not created, skipping cleanup call.")

    print("--> Main script finished.")
    sys.exit(0)