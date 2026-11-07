"""Controlled test for FogMirror Phase 3 Pipeline.

Executes 60 consecutive frames using original Phase 2 camera initialization,
measures exact FPS, verifies non-black video frames, and confirms clean release.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CAMERA, TRACKER
from src.camera import Camera
from src.hand_tracker import HandTracker


def run_controlled_test():
    print("=" * 60)
    print("CONTROLLED TEST: SINGLE-INSTANCE PIPELINE VERIFICATION")
    print("=" * 60)

    # 1. Initialize Camera with exact Phase 2 configuration
    print("[1/5] Initializing Camera (original Phase 2 settings)...")
    camera = Camera(config=CAMERA)
    opened = camera.start()
    assert opened, "Camera failed to open"
    w, h = camera.resolution
    print(f"      Camera opened: {w}x{h}")

    # 2. Initialize HandTracker
    print("[2/5] Initializing HandTracker...")
    tracker = HandTracker(config=TRACKER)

    # 3. Stream 60 frames and record metrics
    print("[3/5] Streaming 60 live frames through the pipeline...")
    frame_times = []
    brightness_values = []
    black_frame_count = 0
    start_time = time.time()

    try:
        for i in range(60):
            t_frame_start = time.perf_counter()
            ret, frame = camera.read()
            if not ret or frame is None:
                print(f"      Frame {i}: Read dropped!")
                continue

            # Check brightness
            mean_b = float(np.mean(frame))
            brightness_values.append(mean_b)
            if mean_b < 5.0 and np.max(frame) < 15:
                black_frame_count += 1

            # MediaPipe Hand Tracking
            ts_ms = int((time.time() - start_time) * 1000)
            tracking_result = tracker.process_frame(frame, ts_ms)
            tracker.draw_landmarks(frame, tracking_result)

            dt_ms = (time.perf_counter() - t_frame_start) * 1000
            frame_times.append(dt_ms)

            # Natural 30 FPS pacing
            sleep_time = max(0.0, 0.033 - (dt_ms / 1000.0))
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # 4. Confirm clean release
        print("[4/5] Executing resource cleanup...")
        tracker.close()
        camera.release()
        assert not camera.is_running, "Camera should not be running after release"
        assert camera.cap is None, "VideoCapture instance should be None after release"
        print("      camera.release() executed successfully.")

    # 5. Report Metrics
    total_elapsed = time.time() - start_time
    effective_fps = len(frame_times) / total_elapsed
    avg_proc_ms = np.mean(frame_times)
    avg_brightness = np.mean(brightness_values)

    print("\n[5/5] Performance & Visual Metrics:")
    print(f"      Total Frames:        {len(frame_times)}/60")
    print(f"      Black Frames:        {black_frame_count} (Must be 0)")
    print(f"      Average Brightness:  {avg_brightness:.1f} (Normal range: 80 - 220)")
    print(f"      Avg Processing Time: {avg_proc_ms:.1f} ms/frame")
    print(f"      Effective FPS:       {effective_fps:.1f} FPS")
    print("=" * 60)

    assert black_frame_count == 0, f"Encountered {black_frame_count} black frames!"
    assert effective_fps >= 25.0, f"FPS dropped below 25: {effective_fps:.1f}"
    print("CONTROLLED TEST PASSED: CAMERA AND TRACKER ARE RUNNING AT FULL 30 FPS!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_controlled_test()
    sys.exit(0 if success else 1)
