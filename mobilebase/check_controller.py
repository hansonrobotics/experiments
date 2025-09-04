import pygame
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick found.")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Joystick: {joystick.get_name()}")
print(f"Buttons: {joystick.get_numbuttons()}")
print(f"Axes: {joystick.get_numaxes()}")

try:
    while True:
        pygame.event.pump()
        
        # Print button states
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        print(f"Buttons: {buttons}", end=" | ")
        
        # Print main axis
        axis_val = joystick.get_axis(1)
        print(f"Axis 1: {axis_val:.2f}")
        
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDone.")
    pygame.quit()
