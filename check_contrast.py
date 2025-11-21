#!/usr/bin/env python3
"""
Camera Contrast Checker - Diagnostic tool for Dr. Strange Shield
Tests camera contrast and hand detection visibility
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse

# Arguments
parser = argparse.ArgumentParser(description='Check camera contrast and hand detection')
parser.add_argument('-c', '--camera', type=int, default=1, help='Camera ID')
parser.add_argument('--no_enhance', action='store_true', help='Disable enhancement')
parser.add_argument('--brightness', type=int, default=30, help='Brightness boost (0-100)')
parser.add_argument('--contrast', type=float, default=2.0, help='CLAHE clip limit (1.0-4.0)')
args = parser.parse_args()


def enhance_frame(frame, brightness_boost=30, clahe_clip=2.0):
    """Enhanced frame with adjustable parameters"""
    try:
        # HSV brightness adjustment
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, brightness_boost)
        v = np.clip(v, 0, 255)
        enhanced_hsv = cv2.merge([h, s, v])
        frame = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        # CLAHE contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return frame
    except Exception as e:
        print(f"Enhancement error: {e}")
        return frame


def calculate_contrast_metrics(frame):
    """Calculate contrast and brightness metrics"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    mean_brightness = np.mean(gray)
    std_dev = np.std(gray)
    min_val = np.min(gray)
    max_val = np.max(gray)
    contrast_ratio = (max_val - min_val) / (max_val + min_val + 1e-6)
    
    return {
        'mean_brightness': mean_brightness,
        'std_dev': std_dev,
        'min': min_val,
        'max': max_val,
        'contrast_ratio': contrast_ratio
    }


def draw_metrics(frame, metrics, hands_detected, enhancement_on):
    """Draw metrics overlay on frame"""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 220), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Title
    title = "CONTRAST DIAGNOSTIC" + (" [ENHANCED]" if enhancement_on else " [RAW]")
    cv2.putText(frame, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Metrics
    y = 65
    line_height = 25
    
    metrics_text = [
        f"Mean Brightness: {metrics['mean_brightness']:.1f} / 255",
        f"Std Deviation: {metrics['std_dev']:.1f}",
        f"Range: {metrics['min']:.0f} - {metrics['max']:.0f}",
        f"Contrast Ratio: {metrics['contrast_ratio']:.3f}",
        f"Hands Detected: {'YES' if hands_detected else 'NO'}"
    ]
    
    for text in metrics_text:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_height
    
    # Quality assessment
    y += 10
    quality = "GOOD" if metrics['mean_brightness'] > 80 and metrics['contrast_ratio'] > 0.3 else "POOR"
    color = (0, 255, 0) if quality == "GOOD" else (0, 0, 255)
    cv2.putText(frame, f"Quality: {quality}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame


# Initialize camera
print("Opening camera...")
cap = cv2.VideoCapture(args.camera)

# Camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Brightness optimization
try:
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    cap.set(cv2.CAP_PROP_GAIN, 10)
except:
    pass

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

print("\n" + "="*60)
print("CAMERA CONTRAST CHECKER")
print("="*60)
print(f"Camera: {args.camera}")
print(f"Enhancement: {'OFF' if args.no_enhance else 'ON'}")
print(f"Brightness boost: {args.brightness}")
print(f"CLAHE clip limit: {args.contrast}")
print("-"*60)
print("Controls:")
print("  Q - Quit")
print("  E - Toggle enhancement")
print("  + - Increase brightness")
print("  - - Decrease brightness")
print("  [ - Decrease contrast")
print("  ] - Increase contrast")
print("="*60 + "\n")

enhancement_on = not args.no_enhance
brightness_boost = args.brightness
clahe_clip = args.contrast

with mp_holistic.Holistic(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5,
    model_complexity=0,
    smooth_landmarks=True
) as holistic:
    
    cv2.namedWindow("Contrast Checker", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Mirror for natural view
        frame = cv2.flip(frame, 1)
        
        # Store original for comparison
        original = frame.copy()
        
        # Apply enhancement if enabled
        if enhancement_on:
            frame = enhance_frame(frame, brightness_boost, clahe_clip)
        
        # Calculate metrics
        metrics = calculate_contrast_metrics(frame)
        
        # Detect hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        hands_detected = False
        if results.left_hand_landmarks or results.right_hand_landmarks:
            hands_detected = True
            
            # Draw hand landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
        
        # Draw metrics overlay
        frame = draw_metrics(frame, metrics, hands_detected, enhancement_on)
        
        # Show frame
        cv2.imshow("Contrast Checker", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('e'):
            enhancement_on = not enhancement_on
            print(f"Enhancement: {'ON' if enhancement_on else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            brightness_boost = min(100, brightness_boost + 5)
            print(f"Brightness boost: {brightness_boost}")
        elif key == ord('-') or key == ord('_'):
            brightness_boost = max(0, brightness_boost - 5)
            print(f"Brightness boost: {brightness_boost}")
        elif key == ord('['):
            clahe_clip = max(1.0, clahe_clip - 0.5)
            print(f"CLAHE clip limit: {clahe_clip}")
        elif key == ord(']'):
            clahe_clip = min(4.0, clahe_clip + 0.5)
            print(f"CLAHE clip limit: {clahe_clip}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("Contrast check complete!")
print("="*60)
