import cv2
import numpy as np
import time
from config import config

# -------------------- Loading Screen UI --------------------
def create_loading_screen(width, height, step, progress):
    """Create loading screen with progress bar"""
    screen = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Use config colors
    start_color = config.COLOR_LOADING_BG_START
    
    for y in range(height):
        # Create gradient
        ratio = 1 - y / height
        color_val = int(30 * ratio)
        screen[y, :] = [
            start_color[0] + color_val, 
            start_color[1] + color_val, 
            start_color[2] + color_val
        ]
    
    np.random.seed(42)
    for _ in range(100):
        cv2.circle(screen, (np.random.randint(0, width), np.random.randint(0, height)), 1, 
                   (np.random.randint(100, 255),)*3, -1)
    
    # Title
    title, font = "DR. STRANGE SHIELDS", cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(title, font, 2.0, 4)
    tx, ty = (width - tw) // 2, height // 3
    cv2.putText(screen, title, (tx + 3, ty + 3), font, 2.0, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(screen, title, (tx, ty), font, 2.0, config.COLOR_HUD_ACCENT, 4, cv2.LINE_AA)
    cv2.putText(screen, "Exhibition Mode", ((width - cv2.getTextSize("Exhibition Mode", font, 0.9, 2)[0][0]) // 2, ty + 50), 
                font, 0.9, (180, 180, 220), 2, cv2.LINE_AA)
    
    # Progress bar
    bw, bh, bx, by = 500, 30, (width - 500) // 2, height // 2 + 50
    cv2.rectangle(screen, (bx, by), (bx + bw, by + bh), config.COLOR_LOADING_BAR_BG, -1)
    cv2.rectangle(screen, (bx, by), (bx + bw, by + bh), (120, 120, 140), 2)
    
    fill = int(bw * (progress / 100))
    for i in range(fill):
        r = i / max(1, fill)
        cv2.line(screen, (bx + i, by + 2), (bx + i, by + bh - 2), 
                (int(100 + 155 * r), int(150 + 100 * r), int(255 * r)), 1)
    
    # Text overlays
    for text, y, scale, thick, color in [
        (f"{int(progress)}%", by + bh + 35, 0.7, 2, (200, 200, 255)),
        (step, by - 25, 0.65, 1, (180, 220, 255)),
        ("Initializing camera and effects...", height - 30, 0.5, 1, (140, 140, 160))
    ]:
        w = cv2.getTextSize(text, font, scale, thick)[0][0]
        cv2.putText(screen, text, ((width - w) // 2, y), font, scale, color, thick, cv2.LINE_AA)
    
    # Spinner
    cx, cy, rad = width // 2, height - 100, 20
    angle = (time.time() * 200) % 360
    for offset in range(0, 270, 30):
        a = (angle + offset) % 360
        alpha = 1.0 - (offset / 270)
        x1 = int(cx + rad * np.cos(np.radians(a)))
        y1 = int(cy + rad * np.sin(np.radians(a)))
        cv2.circle(screen, (x1, y1), 3, (int(255 * alpha), int(127 * alpha), 0), -1)
    
    return screen

# -------------------- Image Enhancement --------------------
def enhance_frame(frame, boost=40):
    """Simple brightness boost for better visibility"""
    # Add brightness boost - adjustable via UI
    brightened = cv2.convertScaleAbs(frame, alpha=1.0, beta=boost)
    return brightened

def analyze_frame_brightness(frame):
    """Analyze frame and calculate optimal brightness boost"""
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness
    avg_brightness = np.mean(gray)
    
    # Optimal brightness target: 120-140 (mid-range)
    
    # Calculate needed boost
    if avg_brightness < 80:  # Very dark
        optimal_boost = 60
    elif avg_brightness < 110:  # Dark
        optimal_boost = 45
    elif avg_brightness < 140:  # Slightly dark
        optimal_boost = 30
    elif avg_brightness < 170:  # Good
        optimal_boost = 15
    else:  # Bright
        optimal_boost = 0
    
    return int(optimal_boost)

def auto_adjust_brightness(frame, current_boost, last_update_time):
    """Automatically adjust brightness based on frame analysis"""
    current_time = time.time()
    # Update every 2 seconds to avoid flickering
    if current_time - last_update_time > 2.0:
        optimal_boost = analyze_frame_brightness(frame)
        # Smooth transition: move towards optimal gradually
        new_boost = int(0.7 * current_boost + 0.3 * optimal_boost)
        return new_boost, current_time
    return current_boost, last_update_time

# -------------------- Exhibition UI Functions --------------------
def draw_distance_feedback(frame, status):
    """Draw feedback if user is too close or too far"""
    if status == "OK":
        return frame
        
    h, w = frame.shape[:2]
    text = ""
    icon = ""
    
    if status == "TOO_CLOSE":
        text = "TOO CLOSE! STEP BACK"
        icon = "⚠️"
    elif status == "TOO_FAR":
        text = "TOO FAR! STEP CLOSER"
        icon = "🔍"
        
    # Draw warning box at top center
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thick = 2
    
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    
    cx = w // 2
    cy = 100
    
    padding = 20
    x1 = cx - tw // 2 - padding
    y1 = cy - th - padding
    x2 = cx + tw // 2 + padding
    y2 = cy + padding
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_DISTANCE_WARNING, 2)
    
    # Text
    cv2.putText(frame, f"{icon} {text}", (x1 + padding, cy), 
                font, scale, config.COLOR_DISTANCE_WARNING, thick, cv2.LINE_AA)
                
    return frame

def draw_instruction_text(frame, text, y_position=None):
    """Draw large centered instruction text"""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Center position
    x = (w - text_w) // 2
    y = y_position if y_position else h // 4
    
    # Draw background rectangle
    padding = 20
    cv2.rectangle(frame, (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + padding), 
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + padding), 
                  config.COLOR_HUD_MAIN, 2)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, config.COLOR_HUD_MAIN, thickness, cv2.LINE_AA)
    return frame

def draw_hand_guides(frame, left_detected=False, right_detected=False):
    """Draw visual guides showing where to place hands"""
    h, w = frame.shape[:2]
    
    # Guide positions (left and right thirds of screen)
    left_x = w // 4
    right_x = 3 * w // 4
    guide_y = h // 2
    guide_size = 80
    
    # Left hand guide
    left_color = config.COLOR_SUCCESS if left_detected else (0, 100, 255)
    cv2.circle(frame, (left_x, guide_y), guide_size, left_color, 3)
    cv2.putText(frame, "LEFT HAND", (left_x - 60, guide_y + guide_size + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2, cv2.LINE_AA)
    
    # Right hand guide
    right_color = config.COLOR_SUCCESS if right_detected else (0, 100, 255)
    cv2.circle(frame, (right_x, guide_y), guide_size, right_color, 3)
    cv2.putText(frame, "RIGHT HAND", (right_x - 70, guide_y + guide_size + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2, cv2.LINE_AA)
    
    return frame

def draw_detection_status(frame, left_detected, right_detected, confidence=0):
    """Draw large detection status indicators"""
    h, w = frame.shape[:2]
    y_pos = h - 100
    
    # Left hand status
    left_text = "✅ LEFT HAND" if left_detected else "❌ LEFT HAND"
    left_color = config.COLOR_SUCCESS if left_detected else config.COLOR_WARNING
    cv2.putText(frame, left_text, (50, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2, cv2.LINE_AA)
    
    # Right hand status
    right_text = "✅ RIGHT HAND" if right_detected else "❌ RIGHT HAND"
    right_color = config.COLOR_SUCCESS if right_detected else config.COLOR_WARNING
    cv2.putText(frame, right_text, (w - 300, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2, cv2.LINE_AA)
    
    return frame

def draw_help_overlay(frame):
    """Draw help overlay with instructions"""
    h, w = frame.shape[:2]
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//4, h//6), (3*w//4, 5*h//6), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "MYSTIC VISION - HELP", (w//4 + 50, h//6 + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, config.COLOR_HUD_MAIN, 2, cv2.LINE_AA)
    
    # Instructions
    instructions = [
        "HOW TO USE:",
        "1. Stand in front of camera",
        "2. Raise BOTH hands",
        "3. Place hands in the circles",
        "4. Keep hands visible and steady",
        "5. Make gesture to activate shields!",
        "",
        "KEYBOARD SHORTCUTS:",
        "A - Toggle Auto-Brightness",
        "H - Toggle this help",
        "G - Toggle shields manually",
        "F - Fullscreen",
        "Q/X - Quit",
        "",
        "Press H to close this help"
    ]
    
    y_start = h//6 + 100
    for i, line in enumerate(instructions):
        y = y_start + i * 35
        color = (255, 255, 255) if line.startswith(("1", "2", "3", "4", "5")) else (200, 200, 200)
        if line.startswith(("HOW", "KEYBOARD")):
            color = config.COLOR_HUD_MAIN
        cv2.putText(frame, line, (w//4 + 70, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1 if line == "" else 2, cv2.LINE_AA)
    
    return frame

def draw_countdown_headline(frame, seconds_left, total_duration=5):
    """Draw a mystical magic circle with futuristic HUD elements - Dr. Strange style"""
    h, w = frame.shape[:2]
    
    # Calculate progress and smooth animations
    progress = (total_duration - seconds_left) / total_duration
    t = time.time()
    
    # HUD-style dimensions
    bar_h = 90
    y = 10
    margin_x = int(w * 0.15)
    bar_w = w - (margin_x * 2)
    
    # Multiple animation speeds
    pulse = (np.sin(t * 2) + 1) / 2 * 0.3 + 0.7
    rotation = (t * 30) % 360  # Slow rotation for magic circles
    fast_rotation = (t * 60) % 360  # Faster rotation
    
    # === LAYER 1: Mystical Background ===
    bg_overlay = frame.copy()
    
    # Dark mystical gradient
    for i in range(bar_h):
        ratio = i / bar_h
        # Deep purple to dark blue gradient
        r_val = int(15 + ratio * 10)
        g_val = int(10 + ratio * 8)
        b_val = int(25 + ratio * 15)
        cv2.line(bg_overlay, (margin_x, y + i), (margin_x + bar_w, y + i), 
                (b_val, g_val, r_val), 1)
    
    cv2.addWeighted(bg_overlay, 0.9, frame, 0.1, 0, frame)
    
    # === LAYER 2: Rotating Magic Circles ===
    circle_overlay = frame.copy()
    center_x = margin_x + bar_w // 2
    center_y = y + bar_h // 2
    
    # Outer magic circle with runes
    outer_radius = 35
    for i in range(12):  # 12 segments like a clock
        angle = rotation + i * 30
        angle_rad = np.radians(angle)
        
        # Draw small circles as "runes"
        rune_x = int(center_x + outer_radius * np.cos(angle_rad))
        rune_y = int(center_y + outer_radius * np.sin(angle_rad))
        
        # Color based on urgency
        if seconds_left <= 3:
            rune_color = (100, 120, 255)  # Red mystical
        elif seconds_left <= 5:
            rune_color = (200, 140, 255)  # Purple mystical
        else:
            rune_color = (255, 180, 100)  # Cyan mystical
        
        cv2.circle(circle_overlay, (rune_x, rune_y), 3, rune_color, -1, cv2.LINE_AA)
    
    # Inner rotating circle (counter-rotation)
    inner_radius = 22
    for i in range(8):
        angle = -fast_rotation + i * 45
        angle_rad = np.radians(angle)
        
        x1 = int(center_x + inner_radius * np.cos(angle_rad))
        y1 = int(center_y + inner_radius * np.sin(angle_rad))
        x2 = int(center_x + (inner_radius - 8) * np.cos(angle_rad))
        y2 = int(center_y + (inner_radius - 8) * np.sin(angle_rad))
        
        if seconds_left <= 3:
            line_color = (120, 140, 255)
        elif seconds_left <= 5:
            line_color = (220, 160, 255)
        else:
            line_color = (255, 200, 120)
        
        cv2.line(circle_overlay, (x1, y1), (x2, y2), line_color, 2, cv2.LINE_AA)
    
    cv2.addWeighted(circle_overlay, 0.5 * pulse, frame, 0.5, 0, frame)
    
    # === LAYER 3: Hexagonal HUD Elements ===
    hex_overlay = frame.copy()
    
    # Left hexagon
    hex_size = 12
    left_hex_x = margin_x + 30
    left_hex_y = y + bar_h // 2
    
    # Right hexagon
    right_hex_x = margin_x + bar_w - 30
    right_hex_y = y + bar_h // 2
    
    for hex_x, hex_y in [(left_hex_x, left_hex_y), (right_hex_x, right_hex_y)]:
        hex_points = []
        for i in range(6):
            angle = rotation + i * 60
            angle_rad = np.radians(angle)
            px = int(hex_x + hex_size * np.cos(angle_rad))
            py = int(hex_y + hex_size * np.sin(angle_rad))
            hex_points.append([px, py])
        
        hex_points = np.array(hex_points, np.int32)
        cv2.polylines(hex_overlay, [hex_points], True, (150, 150, 200), 2, cv2.LINE_AA)
    
    cv2.addWeighted(hex_overlay, 0.4, frame, 0.6, 0, frame)
    
    # === LAYER 4: Futuristic Progress Bar ===
    progress_h = 6
    progress_y = y + bar_h - 18
    progress_w = int(bar_w * progress)
    
    if progress_w > 0:
        # HUD-style segmented progress bar
        segment_count = 40
        segment_width = bar_w / segment_count
        
        for i in range(segment_count):
            seg_x = margin_x + int(i * segment_width)
            seg_w = int(segment_width - 2)
            
            if seg_x + seg_w <= margin_x + progress_w:
                # Filled segment
                ratio = i / segment_count
                
                # Gradient color
                if seconds_left <= 3:
                    color = (100 + int(ratio * 50), 120 + int(ratio * 30), 255)
                elif seconds_left <= 5:
                    color = (200 + int(ratio * 40), 140 + int(ratio * 40), 255)
                else:
                    color = (255, 180 + int(ratio * 40), 100 + int(ratio * 50))
                
                cv2.rectangle(frame, (seg_x, progress_y), 
                            (seg_x + seg_w, progress_y + progress_h), 
                            color, -1, cv2.LINE_AA)
        
        # Glow at progress tip
        glow_overlay = frame.copy()
        tip_x = margin_x + progress_w
        cv2.circle(glow_overlay, (tip_x, progress_y + progress_h // 2), 8, 
                  (255, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay, 0.2 * pulse, frame, 0.8, 0, frame)
    
    # === LAYER 5: Mystical Text with Glow ===
    text = f"NEXT SPELL: {int(seconds_left)}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Pulsing scale for urgency
    if seconds_left <= 3:
        scale_pulse = (np.sin(t * 5) + 1) / 2 * 0.12 + 0.88
        scale = 0.9 * scale_pulse
    else:
        scale = 0.8
    
    thick = 2
    
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = margin_x + (bar_w - tw) // 2
    ty = y + 38
    
    # Mystical color palette
    if seconds_left <= 3:
        text_color = (130, 150, 255)  # Mystical red
        glow_color = (180, 200, 255)
    elif seconds_left <= 5:
        text_color = (220, 160, 255)  # Mystical purple
        glow_color = (255, 200, 255)
    else:
        text_color = (255, 200, 140)  # Mystical cyan-gold
        glow_color = (255, 230, 180)
    
    # Layered shadow for depth
    for offset in [4, 3, 2]:
        shadow_overlay = frame.copy()
        shadow_alpha = 0.3 - offset * 0.05
        cv2.putText(shadow_overlay, text, (tx + offset, ty + offset), 
                   font, scale, (0, 0, 0), thick, cv2.LINE_AA)
        cv2.addWeighted(shadow_overlay, shadow_alpha, frame, 1 - shadow_alpha, 0, frame)
    
    # Mystical glow
    for glow_thick in [6, 4]:
        glow_overlay = frame.copy()
        cv2.putText(glow_overlay, text, (tx, ty), 
                   font, scale, glow_color, glow_thick, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay, 0.1 * pulse, frame, 1 - 0.1 * pulse, 0, frame)
    
    # Main text
    cv2.putText(frame, text, (tx, ty), font, scale, text_color, thick, cv2.LINE_AA)
    
    # === LAYER 6: HUD Border with Scan Lines ===
    border_overlay = frame.copy()
    
    # Main border
    if seconds_left <= 3:
        border_color = (120, 140, 255)
    elif seconds_left <= 5:
        border_color = (200, 160, 255)
    else:
        border_color = (255, 180, 120)
    
    cv2.rectangle(border_overlay, (margin_x, y), (margin_x + bar_w, y + bar_h), 
                 border_color, 2, cv2.LINE_AA)
    cv2.addWeighted(border_overlay, 0.6 * pulse, frame, 0.4, 0, frame)
    
    # Corner brackets (HUD style)
    corner_size = 15
    corner_thick = 3
    corners = [
        (margin_x, y),
        (margin_x + bar_w, y),
        (margin_x, y + bar_h),
        (margin_x + bar_w, y + bar_h)
    ]
    
    bracket_overlay = frame.copy()
    for idx, (corner_x, corner_y) in enumerate(corners):
        # Determine bracket direction
        if idx == 0:  # Top-left
            cv2.line(bracket_overlay, (corner_x, corner_y), (corner_x + corner_size, corner_y), 
                    border_color, corner_thick, cv2.LINE_AA)
            cv2.line(bracket_overlay, (corner_x, corner_y), (corner_x, corner_y + corner_size), 
                    border_color, corner_thick, cv2.LINE_AA)
        elif idx == 1:  # Top-right
            cv2.line(bracket_overlay, (corner_x, corner_y), (corner_x - corner_size, corner_y), 
                    border_color, corner_thick, cv2.LINE_AA)
            cv2.line(bracket_overlay, (corner_x, corner_y), (corner_x, corner_y + corner_size), 
                    border_color, corner_thick, cv2.LINE_AA)
        elif idx == 2:  # Bottom-left
            cv2.line(bracket_overlay, (corner_x, corner_y), (corner_x + corner_size, corner_y), 
                    border_color, corner_thick, cv2.LINE_AA)
            cv2.line(bracket_overlay, (corner_x, corner_y), (corner_x, corner_y - corner_size), 
                    border_color, corner_thick, cv2.LINE_AA)
        else:  # Bottom-right
            cv2.line(bracket_overlay, (corner_x, corner_y), (corner_x - corner_size, corner_y), 
                    border_color, corner_thick, cv2.LINE_AA)
            cv2.line(bracket_overlay, (corner_x, corner_y), (corner_x, corner_y - corner_size), 
                    border_color, corner_thick, cv2.LINE_AA)
    
    cv2.addWeighted(bracket_overlay, 0.7, frame, 0.3, 0, frame)
    
    # Scan line effect
    scan_y = int(y + ((t * 40) % bar_h))
    scan_overlay = frame.copy()
    cv2.line(scan_overlay, (margin_x, scan_y), (margin_x + bar_w, scan_y), 
            (200, 200, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(scan_overlay, 0.15, frame, 0.85, 0, frame)
    
    return frame

def draw_effect_name(frame, effect_name, opacity=1.0):
    """Draw the current effect name with glassmorphism UI"""
    if not effect_name or opacity <= 0:
        return frame
    
    h, w = frame.shape[:2]
    
    # Remove file extension and clean up name
    display_name = effect_name.replace('.mp4', '').replace('.webm', '').replace('.mkv', '')
    
    # Add emoji based on effect type
    emoji_map = {
        'nano': '⚡',
        'eldritch': '🔮',
        'chaos': '💫',
        'cosmic': '✨',
        'time': '⏰',
        'fel': '🔥',
        'mystic': '🌀',
        'arcane': '💠'
    }
    
    emoji = '🛡️'  # default
    for key, em in emoji_map.items():
        if key in display_name.lower():
            emoji = em
            break
    
    display_text = f"{emoji}  {display_name.upper()}"
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
    
    # Position: bottom-left corner with padding
    padding_x = 30
    padding_y = 30
    box_padding = 20
    
    box_x = padding_x
    box_y = h - padding_y - text_h - box_padding * 2
    box_w = text_w + box_padding * 2
    box_h = text_h + box_padding * 2
    
    # Create overlay for transparency
    overlay = frame.copy()
    
    # Draw glassmorphism background
    # Dark semi-transparent background
    bg_color = (20, 20, 30)
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), bg_color, -1)
    
    # Apply transparency based on opacity
    alpha = 0.85 * opacity
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Animated gradient border
    t = time.time()
    border_thickness = 3
    
    # Create gradient effect with multiple colors
    for i in range(border_thickness):
        # Animate color shift
        hue_shift = (t * 50) % 360
        
        # Calculate color based on position and time
        r_val = int(128 + 127 * np.sin(np.radians(hue_shift)))
        g_val = int(128 + 127 * np.sin(np.radians(hue_shift + 120)))
        b_val = int(128 + 127 * np.sin(np.radians(hue_shift + 240)))
        
        border_color = (b_val, g_val, r_val)
        border_alpha = opacity * (1.0 - i / border_thickness)
        
        border_overlay = frame.copy()
        cv2.rectangle(border_overlay, 
                     (box_x - i, box_y - i), 
                     (box_x + box_w + i, box_y + box_h + i), 
                     border_color, 2)
        cv2.addWeighted(border_overlay, border_alpha, frame, 1 - border_alpha, 0, frame)
    
    # Add glow effect
    glow_overlay = frame.copy()
    glow_size = 8
    glow_color = (255, 200, 100)
    cv2.rectangle(glow_overlay, 
                 (box_x - glow_size, box_y - glow_size), 
                 (box_x + box_w + glow_size, box_y + box_h + glow_size), 
                 glow_color, -1)
    cv2.addWeighted(glow_overlay, 0.1 * opacity, frame, 1 - 0.1 * opacity, 0, frame)
    
    # Draw text with shadow
    text_x = box_x + box_padding
    text_y = box_y + box_padding + text_h
    
    # Shadow
    shadow_offset = 3
    shadow_color = (0, 0, 0)
    shadow_overlay = frame.copy()
    cv2.putText(shadow_overlay, display_text, 
               (text_x + shadow_offset, text_y + shadow_offset), 
               font, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.addWeighted(shadow_overlay, 0.6 * opacity, frame, 1 - 0.6 * opacity, 0, frame)
    
    # Main text with gradient
    text_color = (255, 220, 150)  # Golden color
    text_overlay = frame.copy()
    cv2.putText(text_overlay, display_text, (text_x, text_y), 
               font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.addWeighted(text_overlay, opacity, frame, 1 - opacity, 0, frame)
    
    return frame

def overlay_rgba(background, overlay, x, y, w=None, h=None):
    """Optimized RGBA overlay with better performance and smoothness"""
    if overlay is None:
        return background
    
    if w is not None and h is not None and (w > 0 and h > 0):
        try:
            # Use INTER_CUBIC for smoother scaling
            overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_CUBIC)
        except Exception:
            return background
    
    oh, ow = overlay.shape[:2]
    bh, bw = background.shape[:2]
    
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + ow)
    y2 = min(bh, y + oh)
    
    if x2 <= x1 or y2 <= y1:
        return background
    
    ox1 = max(0, -x)
    oy1 = max(0, -y)
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    
    try:
        if overlay.shape[2] == 4:
            overlay_crop = overlay[oy1:oy2, ox1:ox2]
            alpha = overlay_crop[:,:,3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            bg_crop = background[y1:y2, x1:x2]
            fg_crop = overlay_crop[:,:,:3]
            
            blended = (fg_crop * alpha + bg_crop * (1 - alpha)).astype(np.uint8)
            background[y1:y2, x1:x2] = blended
        else:
            background[y1:y2, x1:x2] = overlay[oy1:oy2, ox1:ox2]
    except Exception:
        pass
    
    return background
