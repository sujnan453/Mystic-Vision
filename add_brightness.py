import re

# Read the file
with open('shield.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the enhance_frame function
old_function = '''def enhance_frame(frame):
    """DISABLED - Returns raw camera feed without any processing"""
    return frame'''

new_function = '''def enhance_frame(frame):
    """Simple brightness boost for better visibility"""
    # Add brightness boost - simple and clean
    brightened = cv2.convertScaleAbs(frame, alpha=1.0, beta=40)
    return brightened'''

content = content.replace(old_function, new_function)

# Write back
with open('shield.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Brightness boost added - beta=40 for better visibility")
