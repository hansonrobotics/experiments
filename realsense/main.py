#!/usr/bin/env python3

import rospy
from audio_vision_processor import AudioVisionProcessor

if __name__ == '__main__':
    print("--> Script started.")
    processor = AudioVisionProcessor()
    try:
        print("--> Starting processor.run()... Press Ctrl+C to exit.")
        processor.run()
    except rospy.ROSInterruptException:
        print("\n--> CtrlC detected! Stopping the programme...")
    finally:
        print("--> Reached finally block, performing cleanup.")
        processor.stop()
    print("--> Script finished.")