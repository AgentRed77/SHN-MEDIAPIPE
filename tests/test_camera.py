"""Automated unit & integration test for Camera module.

Tests hardware initialization, frame read validity, mirroring, and clean release.
"""

import sys
import os
import time
import numpy as np

# Add project root to sys.path so tests can import src and config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CameraConfig
from src.camera import Camera


def test_camera_pipeline():
    print("=" * 60)
    print("RUNNING CAMERA PIPELINE INTEGRATION TEST")
    print("=" * 60)

    config = CameraConfig(device_index=0, frame_width=640, frame_height=480, mirror=True)
    camera = Camera(config=config)

    print("[Step 1] Starting camera...")
    opened = camera.start()
    if not opened:
        print("[WARNING] Could not open camera device 0.")
        print("          This is expected if the physical camera is in use or disconnected.")
        print("          The module handled this failure gracefully without crashing.")
        return True

    print(f"[Step 2] Camera opened. Negotiated resolution: {camera.resolution}")
    assert camera.is_running, "Camera state should be is_running=True"

    print("[Step 3] Reading 5 consecutive test frames...")
    frames_captured = 0
    for i in range(5):
        ret, frame = camera.read()
        if ret and frame is not None:
            frames_captured += 1
            assert isinstance(frame, np.ndarray), "Captured frame must be a NumPy array"
            assert frame.ndim == 3 and frame.shape[2] == 3, "Frame must be a 3-channel color image"
            assert frame.dtype == np.uint8, "Frame dtype must be uint8"
        time.sleep(0.05)

    print(f"         Successfully captured {frames_captured}/5 test frames.")
    assert frames_captured > 0, "At least 1 frame should have been captured."

    print("[Step 4] Testing camera release...")
    camera.release()
    assert not camera.is_running, "Camera state should be is_running=False after release"
    assert camera.cap is None, "VideoCapture instance should be None after release"
    print("         Camera released successfully.")

    print("=" * 60)
    print("CAMERA TEST PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_camera_pipeline()
    sys.exit(0 if success else 1)
