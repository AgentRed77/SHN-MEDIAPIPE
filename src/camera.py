"""Camera Module for FogMirror.

Provides a robust, modular wrapper around OpenCV's VideoCapture for reading,
mirroring, and safely releasing the webcam stream.
"""

import sys
from typing import Optional, Tuple
import cv2
import numpy as np

# Handle relative or package imports
try:
    from config import CameraConfig, CAMERA
except ImportError:
    from ..config import CameraConfig, CAMERA


class Camera:
    """Manages the webcam video stream.
    
    Attributes:
        config (CameraConfig): Configuration settings for the camera.
        cap (cv2.VideoCapture): Underlying OpenCV video capture instance.
        is_running (bool): Tracks whether camera capture is active.
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        """Initialize camera settings without immediately opening the hardware.
        
        Args:
            config: Optional CameraConfig instance. If None, default CAMERA is used.
        """
        self.config = config or CAMERA
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self._actual_width: int = 0
        self._actual_height: int = 0

    def start(self) -> bool:
        """Opens the webcam and configures video stream parameters.
        
        Attempts opening the camera with DirectShow backend (fastest on Windows),
        falling back to default backend if needed.
        
        Returns:
            bool: True if camera successfully initialized and opened, False otherwise.
        """
        device_id = self.config.device_index

        # On Windows, cv2.CAP_DSHOW provides fast startup without MSMF delay
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(device_id)

        # Fallback to default backend if DirectShow fails to open
        if not self.cap or not self.cap.isOpened():
            print(f"[Camera] DirectShow initialization failed for device {device_id}. Retrying with default backend...")
            self.cap = cv2.VideoCapture(device_id)

        if not self.cap.isOpened():
            print(f"[Camera] ERROR: Could not open video device at index {device_id}.")
            print("         Please check if another application is using the camera or if permissions are granted.")
            self.is_running = False
            return False

        # Request preferred resolution & FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        # Query actual negotiated hardware resolution
        self._actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"[Camera] Stream initialized successfully:")
        print(f"         Resolution: {self._actual_width}x{self._actual_height} @ {actual_fps:.1f} FPS")
        print(f"         Mirroring: {'Enabled' if self.config.mirror else 'Disabled'}")

        self.is_running = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame from the camera stream.
        
        If mirror configuration is True, flips the frame horizontally so that
        user movement mimics looking directly into a mirror.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (True, frame) on success, (False, None) on failure.
        """
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # cv2.flip(frame, 1) flips horizontally across the Y-axis
        if self.config.mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    @property
    def resolution(self) -> Tuple[int, int]:
        """Returns the actual resolution as (width, height)."""
        return self._actual_width, self._actual_height

    def release(self):
        """Safely stops video capture and releases hardware resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        print("[Camera] Hardware stream released cleanly.")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit ensures guaranteed cleanup."""
        self.release()
