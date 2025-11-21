# shield.py - OPTIMIZED VERSION WITH LOADING UI (COMPLETE FIXED VERSION)
# Dr. Strange shields — smoother, lag-free exhibition version with startup UI

import cv2
import time
import mediapipe as mp
import numpy as np
import os
from utils import mediapipe_detection, get_center_lh, get_center_rh, points_detection, points_detection_hands
from argparse import ArgumentParser
import pickle
from datetime import datetime, timedelta
import pyvirtualcam
from pyvirtualcam import PixelFormat
import signal
import sys
import threading
import queue
from collections import deque


# ----------------- Arguments ----------------- #
parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="ML_model", default='models/model_svm.sav',
                    help="PATH of model FILE.", metavar="FILE")
parser.add_argument("-t", "--threshold", dest="threshold_prediction", default=0.9, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-dc", "--det_conf", dest="min_detection_confidence", default=0.4, type=float,
                    help="Lower threshold for better detection of kids' hands")
parser.add_argument("-tc", "--trk_conf", dest="min_tracking_confidence", default=0.5, type=float)
parser.add_argument("-c", "--camera_id", dest="camera", default=0, type=int)
parser.add_argument("-s", "--shield", dest="shield_video", default='effects/shield.mp4')
parser.add_argument("-o", "--output", dest="output_mode", default='both',
                    choices=['window', 'virtual', 'both'])
parser.add_argument("--start_fullscreen", dest="start_fullscreen", action="store_true")
parser.add_argument("--demo", dest="demo_mode", action="store_true",
                    help="Enable demo/easy mode for kids (looser thresholds & longer windows)")
parser.add_argument("--pred_skip", dest="pred_every_n", default=2, type=int,
                    help="Run prediction every N frames (1 = every frame, 2 = every 2nd frame...)")
parser.add_argument("--mirror", dest="mirror", action="store_true",
                    help="Mirror camera horizontally. Default is OFF (no mirroring).")
parser.add_argument("--max_hands", dest="max_hands", default=6, type=int,
                    help="Maximum number of hands to track.")
parser.add_argument("--preload", dest="preload_count", default=3, type=int,
                    help="Number of effect files to preload at startup.")
parser.add_argument("--buffer_secs", dest="buffer_seconds", default=3.0, type=float,
                    help="Seconds of animation to prebuffer per effect loader.")
parser.add_argument("--mp_skip", dest="mp_every_n", default=1, type=int,
                    help="Run MediaPipe detection every N frames.")
parser.add_argument("--no_enhance", dest="no_enhance", action="store_true",
                    help="Disable image enhancement (brightness/contrast). Default is enhancement ON.")
args = parser.parse_args()
# --------------------------------------------- #


# Globals for cleanup
cap = None
cam = None
show_window = False
stop_threads = False
initialization_complete = False
init_status = {"step": "Starting...", "progress": 0}
init_lock = threading.Lock()

# UI Controls
brightness_boost = 0  # Default brightness boost (0-100) - start at 0, user can adjust
show_controls = True  # Show/hide UI controls
auto_brightness_enabled = False  # Auto-adjust brightness based on frame analysis
last_auto_brightness_time = 0  # Throttle auto-brightness updates

# Tutorial and Exhibition Mode
tutorial_enabled = True  # Enable tutorial instructions
show_help_overlay = False  # Toggle help overlay (press H)
last_activity_time = None  # Track user activity
AUTO_RESET_SECONDS = 30  # Auto-reset after inactivity
session_state = "WELCOME"  # WELCOME, DETECTING, ACTIVE, SUCCESS


def signal_handler(sig, frame):
    global stop_threads
    print("\n\n" + "="*60 + "\n🛑 Interruption received (Ctrl+C)\n🧹 Cleaning resources...")
    stop_threads = True
    for obj in [cap, cam] + list(shield_loaders.values()):
        try:
            obj.release() if hasattr(obj, 'release') else (obj.close() if hasattr(obj, 'close') else obj.stop())
        except: pass
    try:
        if show_window: cv2.destroyAllWindows()
    except: pass
    print("\n🏁 Application terminated\n" + "="*60)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")


# -------------------- Loading Screen UI --------------------
def create_loading_screen(width, height, step, progress):
    """Create loading screen with progress bar"""
    screen = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        color_val = int(30 * (1 - y / height))
        screen[y, :] = [color_val + 20, color_val + 10, color_val]
    
    np.random.seed(42)
    for _ in range(100):
        cv2.circle(screen, (np.random.randint(0, width), np.random.randint(0, height)), 1, 
                   (np.random.randint(100, 255),)*3, -1)
    
    # Title
    title, font = "DR. STRANGE SHIELDS", cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(title, font, 2.0, 4)
    tx, ty = (width - tw) // 2, height // 3
    cv2.putText(screen, title, (tx + 3, ty + 3), font, 2.0, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(screen, title, (tx, ty), font, 2.0, (255, 200, 100), 4, cv2.LINE_AA)
    cv2.putText(screen, "Exhibition Mode", ((width - cv2.getTextSize("Exhibition Mode", font, 0.9, 2)[0][0]) // 2, ty + 50), 
                font, 0.9, (180, 180, 220), 2, cv2.LINE_AA)
    
    # Progress bar
    bw, bh, bx, by = 500, 30, (width - 500) // 2, height // 2 + 50
    cv2.rectangle(screen, (bx, by), (bx + bw, by + bh), (60, 60, 80), -1)
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


def update_init_status(step, progress):
    global init_status
    with init_lock:
        init_status = {"step": step, "progress": progress}


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
    
    # Calculate histogram to detect dark/bright regions
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Optimal brightness target: 120-140 (mid-range)
    target_brightness = 130
    
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


def auto_adjust_brightness(frame):
    """Automatically adjust brightness based on frame analysis"""
    global brightness_boost, last_auto_brightness_time
    
    current_time = time.time()
    # Update every 2 seconds to avoid flickering
    if current_time - last_auto_brightness_time > 2.0:
        optimal_boost = analyze_frame_brightness(frame)
        # Smooth transition: move towards optimal gradually
        brightness_boost = int(0.7 * brightness_boost + 0.3 * optimal_boost)
        last_auto_brightness_time = current_time
        return True
    return False


# -------------------- Exhibition UI Functions --------------------
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
                  (100, 200, 255), 2)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, (100, 200, 255), thickness, cv2.LINE_AA)
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
    left_color = (0, 255, 0) if left_detected else (0, 100, 255)
    cv2.circle(frame, (left_x, guide_y), guide_size, left_color, 3)
    cv2.putText(frame, "LEFT HAND", (left_x - 60, guide_y + guide_size + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2, cv2.LINE_AA)
    
    # Right hand guide
    right_color = (0, 255, 0) if right_detected else (0, 100, 255)
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
    left_color = (0, 255, 0) if left_detected else (0, 0, 255)
    cv2.putText(frame, left_text, (50, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2, cv2.LINE_AA)
    
    # Right hand status
    right_text = "✅ RIGHT HAND" if right_detected else "❌ RIGHT HAND"
    right_color = (0, 255, 0) if right_detected else (0, 0, 255)
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
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2, cv2.LINE_AA)
    
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
            color = (100, 200, 255)
        cv2.putText(frame, line, (w//4 + 70, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1 if line == "" else 2, cv2.LINE_AA)
    
    return frame


# -------------------- Camera Capture Thread --------------------
class CameraThread:
    """Dedicated thread for camera capture to prevent blocking"""
    def __init__(self, camera_id=0):
        update_init_status("Opening camera...", 10)
        self.cap = cv2.VideoCapture(camera_id)
        
        update_init_status("Configuring camera...", 15)
        time.sleep(1.2)
        
        # Request high-res for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Reduce buffer to get latest frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Quality settings for laptop webcam - use AUTO settings for best results
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Full auto-exposure ON
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
            # Don't manually set brightness/contrast - let camera auto-adjust
        except Exception:
            # Some cameras may not support all settings
            pass
        
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.warmup_count = 0
        
        # Get actual resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        
        update_init_status("Camera ready", 20)
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        # Allow camera to warm up
        update_init_status("Warming up camera...", 25)
        time.sleep(0.5)
        
    def _run(self):
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Skip first few frames to allow camera to stabilize
                frame_count += 1
                if frame_count > 10:  # Reduced warmup
                    with self.lock:
                        self.frame = frame
            time.sleep(0.001)
            
    def read(self):
        with self.lock:
            return self.frame is not None, self.frame.copy() if self.frame is not None else None
            
    def release(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()


# Start camera with threading
update_init_status("Initializing...", 5)
camera_thread = CameraThread(args.camera)
camera_thread.start()
width = camera_thread.width
height = camera_thread.height

update_init_status("Loading ML model...", 30)

# Load model
model = pickle.load(open(current_directory + '/' + args.ML_model, 'rb'))
labels = np.array(model.classes_)

update_init_status("Warming up model...", 35)

# Warm up model
try:
    dummy_len = 63
    try:
        dummy_len = len(points_detection_hands(np.zeros((21,3)))) if callable(points_detection_hands) else 63
    except Exception:
        dummy_len = 63
    dummy = np.zeros((1, dummy_len), dtype=float)
except Exception:
    dummy = np.zeros((1, 63))
for _ in range(3):
    try:
        _ = model.predict_proba(dummy)
    except Exception:
        try:
            _ = model.predict(dummy)
        except Exception:
            break

update_init_status("Model ready", 40)

# State flags
KEY_1 = False
KEY_3 = False
SHIELDS = False

scale = 1.5
mp_holistic = mp.solutions.holistic

update_init_status("Loading effects...", 45)

# EFFECTS folder
effects_folder = os.path.join(current_directory, 'effects')
video_exts = ('.mp4', '.mov', '.webm', '.avi', '.mkv')


def list_effects_files():
    return sorted([f for f in os.listdir(effects_folder) if f.lower().endswith(video_exts)]) if os.path.isdir(effects_folder) else []


effects_files = list_effects_files()

shield_path = args.shield_video
if not os.path.isabs(shield_path):
    shield_path = os.path.join(current_directory, shield_path)
shield_loaded_name = os.path.basename(shield_path) if os.path.exists(shield_path) else (effects_files[0] if effects_files else None)

shield = None

update_init_status("Loading portal effects...", 50)

# Portal assets
portal_img_path = os.path.join(current_directory, 'effects', 'portal.png')
portal_vid_path = os.path.join(current_directory, 'effects', 'portal.mp4')
portal_img = None
portal_cap = None
use_portal_video = False
if os.path.exists(portal_img_path):
    portal_img = cv2.imread(portal_img_path, cv2.IMREAD_UNCHANGED)
elif os.path.exists(portal_vid_path):
    portal_cap = cv2.VideoCapture(portal_vid_path)
    if portal_cap.isOpened():
        use_portal_video = True

black_screen = np.array([0,0,0])

# Portal animation
portal_scale = 0.0
portal_anim_speed = 0.08
portal_active_prev = False

# Demo mode settings (optimized for kids)
if args.demo_mode:
    PRED_CONF_THRESHOLD = 0.65  # More forgiving for kids
    KEY3_REQUIRED_IN_WINDOW = 2
    KEY3_WINDOW_FRAMES = 4
    KEY_SEQUENCE_TIME = 2.5
else:
    PRED_CONF_THRESHOLD = 0.72  # Slightly lower for better detection
    KEY3_REQUIRED_IN_WINDOW = 3
    KEY3_WINDOW_FRAMES = 5
    KEY_SEQUENCE_TIME = 1.5

PRED_EVERY_N_FRAMES = max(1, args.pred_every_n)
MP_EVERY_N_FRAMES = max(1, args.mp_every_n)

_recent_preds = []
_frame_count = 0
_mp_frame_count = 0

KEY1_DISPLAY_DURATION = 1.2
key1_display_until = None
key1_last_pos = None

features_q = queue.Queue(maxsize=1)
pred_lock = threading.Lock()
latest_prediction = None
latest_pred_prob = 0.0

# UI elements
BUTTON_W, BUTTON_H = 260, 120
BUTTON_MARGIN = 20
button_tl = (width - BUTTON_W - BUTTON_MARGIN, height - BUTTON_H - BUTTON_MARGIN)
button_br = (width - BUTTON_MARGIN, height - BUTTON_MARGIN)
button_color_bg = (30, 30, 30)
button_color_fg = (200, 200, 255)

SELECT_BTN_W, SELECT_BTN_H = 260, 72
SELECT_BTN_MARGIN = 18
select_btn_tl = (width - SELECT_BTN_W - SELECT_BTN_MARGIN, SELECT_BTN_MARGIN)
select_btn_br = (width - SELECT_BTN_MARGIN, SELECT_BTN_MARGIN + SELECT_BTN_H)
select_btn_color_bg = (40, 40, 60)
select_btn_color_fg = (220, 220, 180)
select_btn_label = "Select shield..."

menu_open = False
selected_index = 0
if effects_files:
    if shield_loaded_name and shield_loaded_name in effects_files:
        selected_index = effects_files.index(shield_loaded_name)
    else:
        selected_index = 0
else:
    selected_index = 0


# -------------------- OPTIMIZED ShieldLoader --------------------
class ShieldLoader:
    """Enhanced loader with better buffering and performance"""
    def __init__(self, path, buffer_seconds=3.0, min_fps=15, max_frames=300):
        self.path = path
        self.buffer_seconds = float(buffer_seconds)
        self.min_fps = min_fps
        self.max_frames = int(max_frames)
        self.cap = None
        self.buffer = deque()
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.fps = None
        self.w = None
        self.h = None
        self.last_frame = None
        self.ready = False
        self.target_initial = 10
        self.target_maintain = 20
        
    def _open(self):
        try:
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap or not self.cap.isOpened():
                return False
            
            # Enable hardware acceleration
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.fps = max(self.min_fps, int(round(fps)))
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            
            target = int(self.buffer_seconds * self.fps)
            target = max(10, min(target, self.max_frames))
            self.target_initial = target
            self.target_maintain = int(target * 1.5)
            
            return True
        except Exception:
            return False

    def start(self):
        if self.running:
            return
        if not self._open():
            self.ready = False
            self.running = False
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        try:
            if self.thread:
                self.thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        with self.lock:
            self.buffer.clear()
        self.ready = False

    def _run(self):
        try:
            filled = 0
            while self.running and filled < self.target_initial:
                ret, f = self.cap.read()
                if not ret or f is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                f = np.ascontiguousarray(f)
                with self.lock:
                    self.buffer.append(f)
                    self.last_frame = f
                    if len(self.buffer) > self.max_frames:
                        self.buffer.popleft()
                filled += 1
            
            self.ready = True
            
            while self.running:
                with self.lock:
                    buflen = len(self.buffer)
                
                if buflen >= self.target_maintain:
                    time.sleep(0.02)
                    continue
                elif buflen > self.target_initial:
                    time.sleep(0.005)
                else:
                    time.sleep(0.001)
                
                ret, f = self.cap.read()
                if not ret or f is None:
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception:
                        pass
                    continue
                
                f = np.ascontiguousarray(f)
                with self.lock:
                    self.buffer.append(f)
                    self.last_frame = f
                    if len(self.buffer) > self.max_frames:
                        self.buffer.popleft()
                        
        except Exception:
            self.ready = False
            self.running = False

    def get_frame(self):
        with self.lock:
            if self.buffer:
                f = self.buffer.popleft()
                self.last_frame = f
                return f
            else:
                return self.last_frame if self.last_frame is not None else None


# Loader registry
shield_loaders = {}
DEFAULT_PREBUFFER_SECONDS = float(args.buffer_seconds)
MAX_FRAMES_PER_LOADER = 300


def start_loader_for_filename(name):
    if not name:
        return None
    fp = name if os.path.isabs(name) else os.path.join(effects_folder, name)
    if not os.path.exists(fp):
        return None
    if name in shield_loaders:
        if not shield_loaders[name].running:
            shield_loaders[name].start()
        return shield_loaders[name]
    loader = ShieldLoader(fp, buffer_seconds=DEFAULT_PREBUFFER_SECONDS, max_frames=MAX_FRAMES_PER_LOADER)
    shield_loaders[name] = loader
    loader.start()
    return loader


# Preload first N effects
PRELOAD_FIRST_N = max(1, min(args.preload_count, len(effects_files)))
for i, fname in enumerate(effects_files[:PRELOAD_FIRST_N]):
    progress = 55 + int((i / PRELOAD_FIRST_N) * 30)
    update_init_status(f"Preloading effect {i+1}/{PRELOAD_FIRST_N}...", progress)
    start_loader_for_filename(fname)
    time.sleep(0.3)

if shield_loaded_name and os.path.basename(shield_loaded_name) in effects_files:
    start_loader_for_filename(os.path.basename(shield_loaded_name))

update_init_status("Finalizing setup...", 90)


def get_shield_frame(selected_name):
    if not selected_name:
        return None, False
    
    loader = shield_loaders.get(selected_name)
    if loader:
        f = loader.get_frame()
        return (f, True) if f is not None else (loader.last_frame, True) if loader.last_frame else (None, False)
    
    fp = selected_name if os.path.isabs(selected_name) else os.path.join(effects_folder, selected_name)
    if os.path.exists(fp):
        try:
            cap_tmp = cv2.VideoCapture(fp)
            ret, f = cap_tmp.read()
            cap_tmp.release()
            if ret and f is not None:
                return np.ascontiguousarray(f), True
        except: pass
    
    return np.zeros((int(height*0.2), int(width*0.2), 3), dtype=np.uint8), False


# -------------------- Inference worker --------------------
def inference_worker():
    global stop_threads, latest_prediction, latest_pred_prob
    while not stop_threads:
        try:
            feats = features_q.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            pred = model.predict(feats)[0]
            prob = float(np.max(model.predict_proba(feats)))
        except Exception:
            pred = None
            prob = 0.0
        with pred_lock:
            latest_prediction = pred
            latest_pred_prob = prob
        time.sleep(0.001)


update_init_status("Starting inference engine...", 93)
inf_thread = threading.Thread(target=inference_worker, daemon=True)
inf_thread.start()


# -------------------- Mouse & UI callbacks --------------------
mouse_clicked = False
def on_mouse(event, x, y, flags, param):
    global SHIELDS, mouse_clicked, menu_open, selected_index, effects_files, shield_loaded_name
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1 = button_tl
        bx2, by2 = button_br
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            SHIELDS = not SHIELDS
            mouse_clicked = True
            return
        
        sx1, sy1 = select_btn_tl
        sx2, sy2 = select_btn_br
        if sx1 <= x <= sx2 and sy1 <= y <= sy2:
            menu_open = not menu_open
            mouse_clicked = True
            return
        
        if menu_open:
            menu_x = sx1
            menu_y = sy2 + 8
            padding = 6
            item_h = 36
            max_items_display = min(12, max(1, len(effects_files)))
            menu_w = SELECT_BTN_W
            menu_h = padding + item_h * max_items_display + padding
            if menu_x <= x <= menu_x + menu_w and menu_y <= y <= menu_y + menu_h:
                rel_y = y - menu_y - padding
                if rel_y < 0:
                    return
                idx = int(rel_y // item_h)
                if 0 <= idx < len(effects_files):
                    chosen = effects_files[idx]
                    selected_index = idx
                    shield_loaded_name = chosen
                    start_loader_for_filename(chosen)
                    print(f"\n🔁 Selected shield file: {chosen}")
                    menu_open = False
                    mouse_clicked = True
                    return
            else:
                menu_open = False
                mouse_clicked = True
                return


show_window = args.output_mode in ['window', 'both']
use_virtual_cam = args.output_mode in ['virtual', 'both']

# Brightness trackbar callback
def on_brightness_change(val):
    global brightness_boost
    brightness_boost = val

print("\n" + "="*60)
print("DR. STRANGE SHIELDS - OPTIMIZED EXHIBITION MODE")
print("="*60)
print(f"Demo mode: {'ON' if args.demo_mode else 'OFF'}")
print(f"Prediction every {PRED_EVERY_N_FRAMES} frame(s)")
print(f"MediaPipe every {MP_EVERY_N_FRAMES} frame(s)")
print(f"Camera mirroring: {'ON' if args.mirror else 'OFF'}")
print(f"Buffer seconds: {args.buffer_seconds}")
print("-" * 60)


# -------------------- OPTIMIZED overlay_rgba function --------------------
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


# -------------------- Loading Screen Display --------------------
update_init_status("Preparing display...", 95)

# Create window and show loading screen
if show_window:
    cv2.namedWindow("Dr. Strange shields", cv2.WINDOW_NORMAL)
    if args.start_fullscreen:
        cv2.setWindowProperty("Dr. Strange shields", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty("Dr. Strange shields", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow("Dr. Strange shields", width, height)
        except Exception:
            pass
    
    # Display loading screen with animation
    loading_start = time.time()
    while time.time() - loading_start < 2.0:
        with init_lock:
            current_step = init_status["step"]
            current_progress = init_status["progress"]
        
        loading_frame = create_loading_screen(width, height, current_step, current_progress)
        cv2.imshow("Dr. Strange shields", loading_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            stop_threads = True
            break
        
        if current_progress < 98:
            update_init_status(current_step, current_progress + 1)

update_init_status("Ready!", 100)

# Final loading screen
if show_window:
    final_loading = create_loading_screen(width, height, "System Ready", 100)
    cv2.imshow("Dr. Strange shields", final_loading)
    cv2.waitKey(500)

initialization_complete = True


# -------------------- Main loop --------------------
cached_mp_results = None
cached_mp_frame = None

target_fps = 60
frame_time = 1.0 / target_fps
last_frame_time = time.time()

with mp_holistic.Holistic(min_detection_confidence=args.min_detection_confidence,
                          min_tracking_confidence=args.min_tracking_confidence,
                          model_complexity=0,  # Fast model for real-time
                          smooth_landmarks=True,  # Smooth hand movements
                          enable_segmentation=False,  # Disable for speed
                          refine_face_landmarks=False) as holistic:  # Disable for speed

    if use_virtual_cam:
        cam = pyvirtualcam.Camera(width, height, target_fps, fmt=PixelFormat.BGR)
        print(f"Virtual cam: {cam.device}")
    else:
        cam = None

    window_created = True
    MINIMIZED_SIZE = (200, 120)

    if show_window:
        cv2.setMouseCallback("Dr. Strange shields", on_mouse)
        # Create brightness control trackbar
        cv2.createTrackbar('Brightness', 'Dr. Strange shields', brightness_boost, 100, on_brightness_change)

    try:
        while camera_thread.running:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            ret, frame = camera_thread.read()
            if not ret or frame is None:
                time.sleep(0.008)
                continue

            frame = np.ascontiguousarray(frame)

            if args.mirror:
                frame = cv2.flip(frame, 1)
            
            # Apply image enhancement if not disabled
            if not args.no_enhance:
                # Auto-adjust brightness if enabled
                if auto_brightness_enabled:
                    auto_adjust_brightness(frame)
                
                frame = enhance_frame(frame, brightness_boost)
                frame = np.ascontiguousarray(frame)

            _frame_count += 1
            _mp_frame_count += 1

            current_shield_name = shield_loaded_name if shield_loaded_name else (effects_files[selected_index] if effects_files else None)
            frame_shield, shield_ok = get_shield_frame(current_shield_name)
            if frame_shield is None:
                frame_shield = np.zeros((200,200,3), dtype=np.uint8)
            else:
                frame_shield = np.ascontiguousarray(frame_shield)

            if _mp_frame_count >= MP_EVERY_N_FRAMES:
                frame_after_mp, results = mediapipe_detection(frame, holistic)
                frame = np.ascontiguousarray(frame_after_mp)
                cached_mp_results = results
                cached_mp_frame = frame.copy()
                _mp_frame_count = 0
            else:
                results = cached_mp_results

            xMinL, xMaxL, yMinL, yMaxL = get_center_lh(frame, results)
            xMinR, xMaxR, yMinR, yMaxR = get_center_rh(frame, results)

            mask = cv2.inRange(frame_shield, black_screen, black_screen)
            res = cv2.bitwise_and(frame_shield, frame_shield, mask=mask)
            res = frame_shield - res
            alpha = 1

            do_predict_this_frame = (_frame_count % PRED_EVERY_N_FRAMES == 0)
            if xMinL and xMinR and do_predict_this_frame:
                try:
                    feats_arr = np.array([points_detection_hands(results)])
                    try:
                        features_q.put_nowait(feats_arr)
                    except queue.Full:
                        try:
                            _ = features_q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            features_q.put_nowait(feats_arr)
                        except queue.Full:
                            pass
                except Exception:
                    pass

            with pred_lock:
                prediction = latest_prediction
                pred_prob = latest_pred_prob

            if SHIELDS:
                torso_center_x = width // 2
                torso_center_y = height // 2
                torso_width = width // 3
                if results and hasattr(results, "pose_landmarks") and results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    try:
                        left_sh = lm[11]
                        right_sh = lm[12]
                        if left_sh.visibility > 0.25 and right_sh.visibility > 0.25:
                            lx, ly = int(left_sh.x * width), int(left_sh.y * height)
                            rx, ry = int(right_sh.x * width), int(right_sh.y * height)
                            torso_center_x = int((lx + rx) / 2)
                            torso_center_y = int((ly + ry) / 2) + 40
                            torso_width = max(80, abs(rx - lx) * 3)
                    except Exception:
                        pass
                portal_w_target = int(torso_width)
                portal_h_target = int(portal_w_target * 1.0)
                if not portal_active_prev:
                    portal_scale = 0.2
                portal_active_prev = True
                portal_scale = min(1.0, portal_scale + portal_anim_speed)
                portal_w = max(1, int(portal_w_target * portal_scale))
                portal_h = max(1, int(portal_h_target * portal_scale))
                top_left_x = int(torso_center_x - portal_w // 2)
                top_left_y = int(torso_center_y - portal_h // 2)
                
                if use_portal_video and portal_cap:
                    ret_p, pframe = portal_cap.read()
                    if not ret_p:
                        portal_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_p, pframe = portal_cap.read()
                    if ret_p and pframe is not None:
                        pframe = np.ascontiguousarray(pframe)
                        try:
                            # Use INTER_CUBIC for smoother portal animation
                            pframe = cv2.resize(pframe, (portal_w, portal_h), interpolation=cv2.INTER_CUBIC)
                        except Exception:
                            pframe = cv2.resize(pframe, (max(1, portal_w), max(1, portal_h)), interpolation=cv2.INTER_LINEAR)
                        gray = cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY)
                        _, alpha_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                        b,g,r = cv2.split(pframe)
                        overlay_rgba_frame = cv2.merge([b,g,r,alpha_mask])
                        frame = overlay_rgba(frame, overlay_rgba_frame, top_left_x, top_left_y)
                elif portal_img is not None:
                    frame = overlay_rgba(frame, portal_img, top_left_x, top_left_y, portal_w, portal_h)
            else:
                portal_active_prev = False
                portal_scale = 0.0

            if SHIELDS and xMinL:
                xc_lh = int(width * ((xMaxL + xMinL) / 2))
                yc_lh = int(height * ((yMaxL + yMinL) / 2))
                l_width_shield = int(width * (xMaxL - xMinL) / 2 * 3.5 * scale)
                l_height_shield = int(height * (yMaxL - yMinL) / 2 * 2 * scale)
                if l_width_shield > 0 and l_height_shield > 0:
                    # Use INTER_CUBIC for smoother shield animation
                    res2 = cv2.resize(res, (max(1, l_width_shield * 2), max(1, l_height_shield * 2)), 
                                     interpolation=cv2.INTER_CUBIC)
                    start_h = 0; start_w = 0
                    stop_h = l_height_shield * 2; stop_w = l_width_shield * 2
                    f_start_h = yc_lh - l_height_shield; f_stop_h = yc_lh + l_height_shield
                    f_start_w = xc_lh - l_width_shield; f_stop_w = xc_lh + l_width_shield
                    
                    if f_start_h < 0:
                        start_h = -f_start_h; f_start_h = 0
                    if f_stop_h > height:
                        stop_h = stop_h - (f_stop_h - height); f_stop_h = height
                    if f_start_w < 0:
                        start_w = -f_start_w; f_start_w = 0
                    if f_stop_w > width:
                        stop_w = stop_w - (f_stop_w - width); f_stop_w = width
                    
                    if stop_h > start_h and stop_w > start_w:
                        res2 = res2[start_h:stop_h, start_w:stop_w, :]
                        try:
                            bg = frame[f_start_h:f_stop_h, f_start_w:f_stop_w]
                            blended = cv2.addWeighted(bg, alpha, res2, 1, 0)
                            frame[f_start_h:f_stop_h, f_start_w:f_stop_w] = blended
                        except Exception:
                            frame[f_start_h:f_stop_h, f_start_w:f_stop_w] = res2[:f_stop_h-f_start_h, :f_stop_w-f_start_w]

            if SHIELDS and xMinR:
                xc_rh = int(width * ((xMaxR + xMinR) / 2))
                yc_rh = int(height * ((yMaxR + yMinR) / 2))
                r_width_shield = int(width * (xMaxR - xMinR) / 2 * 3.5 * scale)
                r_height_shield = int(height * (yMaxR - yMinR) / 2 * 2 * scale)
                if r_width_shield > 0 and r_height_shield > 0:
                    # Use INTER_CUBIC for smoother shield animation
                    res3 = cv2.resize(res, (max(1, r_width_shield * 2), max(1, r_height_shield * 2)),
                                     interpolation=cv2.INTER_CUBIC)
                    start_h = 0; start_w = 0
                    stop_h = r_height_shield * 2; stop_w = r_width_shield * 2
                    f_start_h = yc_rh - r_height_shield; f_stop_h = yc_rh + r_height_shield
                    f_start_w = xc_rh - r_width_shield; f_stop_w = xc_rh + r_width_shield
                    
                    if f_start_h < 0:
                        start_h = -f_start_h; f_start_h = 0
                    if f_stop_h > height:
                        stop_h = stop_h - (f_stop_h - height); f_stop_h = height
                    if f_start_w < 0:
                        start_w = -f_start_w; f_start_w = 0
                    if f_stop_w > width:
                        stop_w = stop_w - (f_stop_w - width); f_stop_w = width
                    
                    if stop_h > start_h and stop_w > start_w:
                        res3 = res3[start_h:stop_h, start_w:stop_w, :]
                        try:
                            bg = frame[f_start_h:f_stop_h, f_start_w:f_stop_w]
                            blended = cv2.addWeighted(bg, alpha, res3, 1, 0)
                            frame[f_start_h:f_stop_h, f_start_w:f_stop_w] = blended
                        except Exception:
                            frame[f_start_h:f_stop_h, f_start_w:f_stop_w] = res3[:f_stop_h-f_start_h, :f_stop_w-f_start_w]

            if xMinL and xMinR:
                use_pred = prediction
                use_prob = pred_prob
                if use_pred is None:
                    try:
                        arr = np.array([points_detection_hands(results)])
                        use_pred = model.predict(arr)[0]
                        use_prob = float(np.max(model.predict_proba(arr)))
                    except Exception:
                        use_pred = None
                        use_prob = 0.0
                
                if SHIELDS:
                    if (use_pred == 'key_4') and (use_prob > PRED_CONF_THRESHOLD):
                        KEY_1 = False; KEY_3 = False; SHIELDS = False
                        _recent_preds = []
                else:
                    if (use_pred == 'key_1') and (use_prob > PRED_CONF_THRESHOLD):
                        t1 = datetime.now()
                        KEY_1 = True
                        _recent_preds = []
                        if xMinL:
                            xc_lh = int(width * ((xMaxL + xMinL) / 2))
                            yc_lh = int(height * ((yMaxL + yMinL) / 2))
                            key1_display_until = datetime.now() + timedelta(seconds=KEY1_DISPLAY_DURATION)
                            key1_last_pos = (xc_lh, yc_lh)
                    elif KEY_1:
                        is_key3_frame = (use_pred == 'key_3') and (use_prob > PRED_CONF_THRESHOLD)
                        _recent_preds.append(1 if is_key3_frame else 0)
                        if len(_recent_preds) > KEY3_WINDOW_FRAMES:
                            _recent_preds.pop(0)
                        key3_pos_count = sum(_recent_preds)
                        if t1 is not None and (t1 + timedelta(seconds=KEY_SEQUENCE_TIME) >= datetime.now()):
                            if key3_pos_count >= KEY3_REQUIRED_IN_WINDOW:
                                KEY_3 = True
                                SHIELDS = True
                                _recent_preds = []
                        else:
                            KEY_1 = False
                            _recent_preds = []
                    else:
                        _recent_preds = []
            else:
                _recent_preds = []

            bar_w = 360
            bar_h = 24
            bar_x = 30
            bar_y = height - 60
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180,180,180), 2)
            
            if KEY_1:
                positive = sum(_recent_preds)
                denom = max(1, KEY3_WINDOW_FRAMES)
                fill = int((positive / denom) * (bar_w - 4))
                cv2.rectangle(frame, (bar_x+2, bar_y+2), (bar_x+2+fill, bar_y+bar_h-2), (0,200,80), -1)
                cv2.putText(frame, "Waiting for step 2...", (bar_x, bar_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Step 1: Raise both hands", (bar_x, bar_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

            bx1, by1 = button_tl
            bx2, by2 = button_br
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), button_color_bg, -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (120,120,120), 2)
            btn_text = "TOGGLE SHIELDS (STAFF)"
            cv2.putText(frame, btn_text, (bx1 + 12, by1 + BUTTON_H // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, button_color_fg, 2, cv2.LINE_AA)

            sx1, sy1 = select_btn_tl
            sx2, sy2 = select_btn_br
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), select_btn_color_bg, -1)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (140,140,160), 2)
            txt_x = sx1 + 12
            txt_y = sy1 + SELECT_BTN_H//2 + 8
            cv2.putText(frame, select_btn_label, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, select_btn_color_fg, 2, cv2.LINE_AA)

            if menu_open:
                menu_x = sx1
                menu_y = sy2 + 8
                padding = 6
                item_h = 36
                max_display = min(12, max(1, len(effects_files)))
                menu_w = SELECT_BTN_W
                menu_h = padding + item_h * max_display + padding
                cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_w, menu_y + menu_h), (30,30,40), -1)
                cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_w, menu_y + menu_h), (120,120,140), 2)
                
                for i in range(max_display):
                    y0 = menu_y + padding + i * item_h
                    y1 = y0 + item_h
                    if i < len(effects_files):
                        fname = effects_files[i]
                        color_text = (200,200,200)
                        if fname == shield_loaded_name:
                            cv2.rectangle(frame, (menu_x + 4, y0 + 4), (menu_x + menu_w - 4, y1 - 4), (60,60,80), -1)
                            color_text = (255,230,120)
                        cv2.putText(frame, fname, (menu_x + 8, y0 + item_h//2 + 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1, cv2.LINE_AA)

            if KEY_1 or (key1_display_until and datetime.now() < key1_display_until):
                if key1_last_pos:
                    hx, hy = key1_last_pos
                else:
                    hx, hy = width//2, height//2
                for i, (r, thickness, alpha_val) in enumerate([(60, 6, 0.18), (40, 5, 0.22), (24, 3, 0.35)]):
                    overlay_temp = frame.copy()
                    color = (0, 220, 255)
                    cv2.circle(overlay_temp, (hx, hy), r, color, thickness, cv2.LINE_AA)
                    cv2.addWeighted(overlay_temp, alpha_val, frame, 1 - alpha_val, 0, frame)

            label = shield_loaded_name if shield_loaded_name else (effects_files[selected_index] if effects_files else "none")
            status = f"SHIELDS={'ON' if SHIELDS else 'OFF'} | {label}"
            cv2.putText(frame, status, (width - 700, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)
            
            # Display brightness and accuracy indicators
            if show_controls:
                # Auto-brightness button (top left)
                button_x, button_y, button_w, button_h = 30, 10, 180, 25
                button_color = (0, 200, 100) if auto_brightness_enabled else (100, 100, 100)
                cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), button_color, -1)
                cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), (200, 200, 200), 2)
                button_text = "AUTO" if auto_brightness_enabled else "MANUAL"
                cv2.putText(frame, f"[A] {button_text}", (button_x + 10, button_y + 18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Brightness indicator (below button)
                brightness_text = f"Brightness: {brightness_boost}" + (" (AUTO)" if auto_brightness_enabled else "")
                cv2.putText(frame, brightness_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
                
                # Hand detection confidence (below brightness)
                if xMinL and xMinR:
                    conf_text = f"Detection: {int(pred_prob * 100)}%"
                    conf_color = (0, 255, 0) if pred_prob > 0.7 else (0, 200, 255) if pred_prob > 0.5 else (0, 100, 255)
                    cv2.putText(frame, conf_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2, cv2.LINE_AA)
                    
                    # Accuracy bar
                    bar_x, bar_y, bar_w, bar_h = 30, 90, 200, 15
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
                    fill_w = int(bar_w * pred_prob)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), conf_color, -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
                else:
                    cv2.putText(frame, "Detection: No hands", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)
            
            # Exhibition Mode: Tutorial and Instructions
            if tutorial_enabled and not SHIELDS:
                # Draw instruction based on detection state
                if not xMinL and not xMinR:
                    draw_instruction_text(frame, "STEP 1: RAISE BOTH HANDS")
                    draw_hand_guides(frame, False, False)
                elif xMinL and not xMinR:
                    draw_instruction_text(frame, "STEP 2: RAISE RIGHT HAND TOO")
                    draw_hand_guides(frame, True, False)
                elif not xMinL and xMinR:
                    draw_instruction_text(frame, "STEP 2: RAISE LEFT HAND TOO")
                    draw_hand_guides(frame, False, True)
                else:
                    # Both hands detected
                    draw_instruction_text(frame, "GREAT! KEEP HANDS VISIBLE")
                    draw_hand_guides(frame, True, True)
                
                # Draw detection status at bottom
                draw_detection_status(frame, xMinL is not None, xMinR is not None, pred_prob)
            
            # Help overlay (toggle with H key)
            if show_help_overlay:
                draw_help_overlay(frame)
            
            # Auto-reset logic: track activity
            if xMinL or xMinR:
                last_activity_time = time.time()
            elif last_activity_time and (time.time() - last_activity_time > AUTO_RESET_SECONDS):
                # Reset after inactivity
                SHIELDS = False
                KEY_1 = False
                KEY_3 = False
                last_activity_time = None
                session_state = "WELCOME"

            if show_window:
                cv2.imshow('Dr. Strange shields', frame)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if key == ord('a') or key == ord('A'):
                        auto_brightness_enabled = not auto_brightness_enabled
                        if auto_brightness_enabled:
                            last_auto_brightness_time = 0  # Force immediate update
                    elif key == ord('h') or key == ord('H'):
                        show_help_overlay = not show_help_overlay
                    elif key == ord('g'):
                        SHIELDS = not SHIELDS
                    elif key == ord('f'):
                        cv2.setWindowProperty("Dr. Strange shields", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    elif key == ord('r'):
                        cv2.setWindowProperty("Dr. Strange shields", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    elif key == ord('m'):
                        try:
                            cv2.resizeWindow("Dr. Strange shields", MINIMIZED_SIZE[0], MINIMIZED_SIZE[1])
                        except Exception:
                            pass
                    elif key == ord('M'):
                        cv2.setWindowProperty("Dr. Strange shields", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    elif key == ord('x') or key == ord('q'):
                        break

            if use_virtual_cam and cam:
                cam.send(frame)
                cam.sleep_until_next_frame()
            
            last_frame_time = time.time()
            sleep_time = frame_time - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n🛑 Interruption received - closing...")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
    finally:
        stop_threads = True
        try:
            inf_thread.join(timeout=1.0)
        except Exception:
            pass
        for loader in list(shield_loaders.values()):
            try:
                loader.stop()
            except Exception:
                pass
        print("\n\n" + "="*60)
        print("\n🧹 Final cleanup...\n")
        camera_thread.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if cam:
            try:
                cam.close()
            except Exception:
                pass
        print("\n🏁 Application terminated\n")
