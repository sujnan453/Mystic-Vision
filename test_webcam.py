import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# Set camera properties for better visibility
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Try to set auto-exposure (1 = auto, 0.25 = manual)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure ON
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Auto focus ON

# Brightness trackbar value (0-100)
brightness = 0

def on_brightness_change(val):
    global brightness
    brightness = val

# Create window and trackbar
cv2.namedWindow('Webcam Test - Press Q to quit')
cv2.createTrackbar('Brightness', 'Webcam Test - Press Q to quit', 0, 100, on_brightness_change)

print("Webcam Test Started!")
print("- Use the 'Brightness' slider to adjust brightness")
print("- Press 'Q' to quit")
print("- Press 'R' to reset brightness to 0")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Apply brightness adjustment if needed
    if brightness > 0:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness)
    
    # Display the frame
    cv2.imshow('Webcam Test - Press Q to quit', frame)
    
    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('r') or key == ord('R'):
        cv2.setTrackbarPos('Brightness', 'Webcam Test - Press Q to quit', 0)
        brightness = 0

# Release everything
cap.release()
cv2.destroyAllWindows()
print("Webcam test ended.")
