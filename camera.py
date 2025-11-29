import cv2
import threading
import time
from state import update_init_status
from config import config

class CameraThread:
    """Dedicated thread for camera capture to prevent blocking"""
    def __init__(self, camera_id=None):
        # Use config ID if none provided
        cam_id = camera_id if camera_id is not None else config.CAMERA_ID
        
        update_init_status("Opening camera...", 10)
        self.cap = cv2.VideoCapture(cam_id)
        
        update_init_status("Configuring camera...", 15)
        time.sleep(1.2)
        
        # Request high-res for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        # Reduce buffer to get latest frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.BUFFER_SIZE)
        
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
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.FRAME_WIDTH
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.FRAME_HEIGHT
        
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
