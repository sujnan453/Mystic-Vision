import cv2
import threading
import time
import os
import numpy as np
from collections import deque
from state import update_init_status
from config import config

# Try to import procedural effects if available
try:
    from procedural_effects import RealisticFireShield
    procedural_effects_classes = {"RealisticFireShield": RealisticFireShield}
except ImportError:
    procedural_effects_classes = {}

procedural_effects = {}
shield_loaders = {}

# -------------------- OPTIMIZED ShieldLoader --------------------
class ShieldLoader:
    """Enhanced loader with better buffering, resizing, and speed control"""
    def __init__(self, path, buffer_seconds=None, min_fps=15, max_frames=None, resize_max_width=None, speed_factor=1.0):
        self.path = path
        self.buffer_seconds = float(buffer_seconds) if buffer_seconds is not None else config.EFFECT_BUFFER_SECONDS
        self.min_fps = min_fps
        self.max_frames = int(max_frames) if max_frames is not None else config.EFFECT_MAX_FRAMES
        self.resize_max_width = resize_max_width if resize_max_width is not None else config.EFFECT_RESIZE_WIDTH
        self.speed_factor = float(speed_factor)
        
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
            
            w_orig = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_orig = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if self.resize_max_width and w_orig > self.resize_max_width:
                scale = self.resize_max_width / w_orig
                self.w = int(w_orig * scale)
                self.h = int(h_orig * scale)
            else:
                self.w = w_orig
                self.h = h_orig
            
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
            frame_accumulator = 0.0
            
            while self.running:
                # Pre-fill check
                with self.lock:
                    buflen = len(self.buffer)
                
                if self.ready:
                    if buflen >= self.target_maintain:
                        time.sleep(0.02)
                        continue
                    elif buflen > self.target_initial:
                        time.sleep(0.005)
                    else:
                        time.sleep(0.001)
                elif filled >= self.target_initial:
                    self.ready = True
                    continue

                ret, f = self.cap.read()
                if not ret or f is None:
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception:
                        pass
                    continue
                
                # Speed control logic
                frame_accumulator += self.speed_factor
                if frame_accumulator < 1.0:
                    continue
                
                # If we need to process this frame
                frame_accumulator -= 1.0
                
                # While loop to skip extra frames if speed_factor > 2.0 etc
                while frame_accumulator >= 1.0:
                    ret, _ = self.cap.read() # Skip frame
                    if not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_accumulator -= 1.0

                # Resize if needed
                if self.resize_max_width and f.shape[1] > self.resize_max_width:
                    try:
                        f = cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_AREA)
                    except Exception:
                        pass

                f = np.ascontiguousarray(f)
                with self.lock:
                    self.buffer.append(f)
                    self.last_frame = f
                    if len(self.buffer) > self.max_frames:
                        self.buffer.popleft()
                
                filled += 1
                        
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

def get_shield_frame(name):
    """Return the current shield frame and a status flag. If using procedural, call the effect; if video, get from loader."""
    if name in procedural_effects:
        try:
            frame = procedural_effects[name].render()
            return frame, True
        except Exception:
            return None, False
    elif name in shield_loaders:
        loader = shield_loaders[name]
        if loader.ready:
            frame = loader.get_frame()
            return frame, True
        else:
            return None, False
    else:
        return None, False

def start_loader_for_filename(name, effects_folder, buffer_seconds=None):
    if not name:
        return None
    fp = name if os.path.isabs(name) else os.path.join(effects_folder, name)
    if not os.path.exists(fp):
        return None
    if name in shield_loaders:
        if not shield_loaders[name].running:
            shield_loaders[name].start()
        return shield_loaders[name]
    
    # Custom settings for specific files
    speed = 1.0
    if "nano" in name.lower():
        speed = 1.5  # Play Nano Tech Shield 50% faster
        print(f"🚀 Boosting speed for {name}")
        
    loader = ShieldLoader(fp, 
                          buffer_seconds=buffer_seconds, 
                          max_frames=config.EFFECT_MAX_FRAMES,
                          resize_max_width=config.EFFECT_RESIZE_WIDTH, # Resize to 640px width for performance
                          speed_factor=speed)
    shield_loaders[name] = loader
    loader.start()
    return loader

def list_effects_files(effects_folder):
    video_exts = ('.mp4', '.mov', '.webm', '.avi', '.mkv')
    # Check for video files (backward compatibility)
    video_files = sorted([f for f in os.listdir(effects_folder) if f.lower().endswith(video_exts)]) if os.path.isdir(effects_folder) else []
    # Add procedural effect names
    procedural_names = list(procedural_effects.keys())
    return procedural_names + video_files

def initialize_procedural_effects(width, height):
    """Initialize procedural effects with screen dimensions"""
    for name, EffectClass in procedural_effects_classes.items():
        procedural_effects[name] = EffectClass(width, height)
        # Also add to list of available effects if not already there (though list_effects_files handles keys)
