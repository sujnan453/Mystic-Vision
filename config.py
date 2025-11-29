import os
from dataclasses import dataclass

@dataclass
class Config:
    # Camera Settings
    CAMERA_ID: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    BUFFER_SIZE: int = 1
    MIRROR_CAMERA: bool = True
    
    # Hand Detection Settings
    MAX_HANDS: int = 2
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.6
    MODEL_PATH: str = 'model_svm.sav'
    MP_EVERY_N_FRAMES: int = 2
    PRED_EVERY_N_FRAMES: int = 2
    
    # Distance Feedback (Hand Size relative to screen height)
    MIN_HAND_SIZE: float = 0.15  # Too far if hand height < 15% of screen height
    MAX_HAND_SIZE: float = 0.8   # Too close if hand height > 80% of screen height
    
    # Exhibition Settings
    TUTORIAL_ENABLED: bool = True
    AUTO_RESET_SECONDS: int = 30
    EFFECT_DURATION: int = 8
    AUTO_SWITCH_ENABLED: bool = True
    EFFECT_CYCLE_DURATION: int = 5
    COUNTDOWN_DURATION: int = 5
    
    # UI Settings
    SHOW_CONTROLS: bool = True
    SHOW_FPS: bool = True
    FULLSCREEN: bool = False
    
    # UI Dimensions
    BUTTON_WIDTH: int = 260
    BUTTON_HEIGHT: int = 120
    BUTTON_MARGIN: int = 20
    SELECT_BTN_WIDTH: int = 260
    SELECT_BTN_HEIGHT: int = 72
    SELECT_BTN_MARGIN: int = 18
    
    # Effects Settings
    EFFECT_BUFFER_SECONDS: float = 2.0
    EFFECT_MAX_FRAMES: int = 300
    EFFECT_RESIZE_WIDTH: int = 640
    PRELOAD_COUNT: int = 2
    
    # Paths
    ASSETS_DIR: str = os.path.join(os.path.dirname(__file__), 'assets')
    EFFECTS_DIR: str = os.path.join(os.path.dirname(__file__), 'effects')
    
    # Colors (BGR)
    COLOR_TEXT: tuple = (255, 255, 255)
    COLOR_ACCENT: tuple = (100, 200, 255)
    COLOR_WARNING: tuple = (0, 0, 255)
    COLOR_SUCCESS: tuple = (0, 255, 0)
    COLOR_LOADING_BG_START: tuple = (20, 10, 20) # Dark purple/black
    COLOR_LOADING_BAR_BG: tuple = (60, 60, 80)
    COLOR_HUD_MAIN: tuple = (100, 200, 255)
    COLOR_HUD_ACCENT: tuple = (255, 180, 100)
    COLOR_DISTANCE_WARNING: tuple = (0, 165, 255) # Orange

config = Config()
