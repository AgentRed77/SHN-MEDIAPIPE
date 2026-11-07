"""Configuration settings for FogMirror.

Centralizes all configurable parameters such as camera settings,
window properties, hand tracker parameters, and display configurations.
"""

import os
from dataclasses import dataclass

# Project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


@dataclass
class CameraConfig:
    """Webcam capture configuration parameters."""
    device_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    mirror: bool = True  # Mirror horizontally for natural mirror feel


@dataclass
class TrackerConfig:
    """MediaPipe HandLandmarker configuration parameters."""
    model_path: str = MODEL_PATH
    model_url: str = MODEL_URL
    num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smoothing_alpha: float = 0.65  # Higher = more responsive, lower = smoother


@dataclass
class CanvasConfig:
    """Canvas and wipe mask configuration parameters."""
    brush_radius: int = 24           # Radius of fingertip wipe path in pixels
    feather_radius: int = 11          # Gaussian blur radius for soft wipe edges (must be odd)
    max_interpolation_step: float = 8.0  # Max distance in px between interpolated stroke stamps
    decay_rate: float = 0.0          # Optional natural refogging decay (0.0 = persistent until cleared)


@dataclass
class WindowConfig:
    """Application window configuration parameters."""
    title: str = "FogMirror - Virtual Fog Glass Writing System"
    fullscreen: bool = False


# Default configuration instances
CAMERA = CameraConfig()
TRACKER = TrackerConfig()
CANVAS = CanvasConfig()
WINDOW = WindowConfig()
