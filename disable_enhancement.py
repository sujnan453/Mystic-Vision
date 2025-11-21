import re

# Read the file
with open('shield.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the enhance_frame function
pattern = r'def enhance_frame\(frame\):.*?(?=\n\ndef |\n\n# ----|\Z)'
replacement = '''def enhance_frame(frame):
    """DISABLED - Returns raw camera feed without any processing"""
    return frame
'''

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open('shield.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ enhance_frame() disabled - now returns raw camera feed")
