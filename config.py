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
    
    # Exhibition Settings
    TUTORIAL_ENABLED: bool = True
    AUTO_RESET_SECONDS: int = 30
    EFFECT_DURATION: int = 8
    AUTO_SWITCH_ENABLED: bool = True
    EFFECT_CYCLE_DURATION: int = 5
    
    # UI Settings
    SHOW_CONTROLS: bool = True
    SHOW_FPS: bool = True
    FULLSCREEN: bool = False
    
    # Paths
    ASSETS_DIR: str = os.path.join(os.path.dirname(__file__), 'assets')
    EFFECTS_DIR: str = os.path.join(os.path.dirname(__file__), 'effects')
    
    # Colors (BGR)
    COLOR_TEXT: tuple = (255, 255, 255)
    COLOR_ACCENT: tuple = (100, 200, 255)
    COLOR_WARNING: tuple = (0, 0, 255)
    COLOR_SUCCESS: tuple = (0, 255, 0)

config = Config()
