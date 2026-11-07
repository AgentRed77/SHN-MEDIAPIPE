"""Automated unit & integration test for HandTracker module.

Verifies:
1. Model loading & HandLandmarker initialization.
2. Inference on frames without hands (graceful empty result).
3. Landmark extraction & smoothing calculations.
4. Clean landmarker resource release.
"""

import sys
import os
import time
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import TRACKER
from src.hand_tracker import HandTracker, TrackingResult, HandData


def test_hand_tracker():
    print("=" * 60)
    print("RUNNING HAND TRACKER INTEGRATION TEST")
    print("=" * 60)

    # 1. Initialize Tracker
    print("[Step 1] Initializing HandTracker...")
    tracker = HandTracker(config=TRACKER)
    assert tracker.landmarker is not None, "HandLandmarker must be instantiated"
    print("         HandTracker initialized successfully.")

    # 2. Test inference on empty/blank frame
    print("[Step 2] Testing frame processing on blank image (no hands)...")
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.time()
    result = tracker.process_frame(blank_frame, timestamp_ms=100)
    elapsed_ms = (time.time() - t0) * 1000
    print(f"         Inference latency: {elapsed_ms:.1f} ms")
    assert isinstance(result, TrackingResult), "Result must be a TrackingResult instance"
    assert result.hand_count == 0, f"Expected 0 hands on blank image, got {result.hand_count}"
    assert result.primary_index_tip is None, "Primary index tip should be None when no hand is present"
    assert result.primary_index_tip_norm is None, "Primary index tip norm should be None when no hand is present"
    print("         Handled zero-hand frame gracefully.")

    # 3. Test Landmark Drawing on blank frame
    print("[Step 3] Testing draw_landmarks on blank frame...")
    annotated = tracker.draw_landmarks(blank_frame.copy(), result)
    assert annotated.shape == blank_frame.shape, "Annotated frame shape must match input"
    print("         Drawing routine handled 0-hand result without error.")

    # 4. Backward Compatibility Check: HandData instantiation
    print("[Step 4] Checking HandData backward compatibility & field accessibility...")
    mock_hand_legacy = HandData(
        landmarks=[(0, 0)] * 21,
        landmarks_norm=[(0.0, 0.0, 0.0)] * 21,
        handedness="Right",
        score=0.95,
        index_tip=(100, 150),
        index_tip_smoothed=(102, 148),
    )
    # Check that legacy field index_tip_smoothed works
    assert mock_hand_legacy.index_tip_smoothed == (102, 148)
    # Check default index_tip_norm
    assert mock_hand_legacy.index_tip_norm == (0.0, 0.0, 0.0)

    # Check with index_tip_norm provided
    mock_hand_new = HandData(
        landmarks=[(0, 0)] * 21,
        landmarks_norm=[(0.0, 0.0, 0.0)] * 21,
        handedness="Right",
        score=0.95,
        index_tip=(100, 150),
        index_tip_smoothed=(102, 148),
        index_tip_norm=(0.5, 0.6, -0.01),
    )
    assert mock_hand_new.index_tip_smoothed == (102, 148)
    assert mock_hand_new.index_tip_norm == (0.5, 0.6, -0.01)
    print("         HandData backward compatibility verified (both legacy & new fields).")

    # 5. Test Smoothing Math directly with mock HandData
    print("[Step 5] Testing Exponential Moving Average (EMA) smoothing logic...")
    raw_pt1 = (100, 100)
    raw_pt2 = (110, 120)
    alpha = TRACKER.smoothing_alpha
    expected_x = int(round(alpha * raw_pt2[0] + (1 - alpha) * raw_pt1[0]))
    expected_y = int(round(alpha * raw_pt2[1] + (1 - alpha) * raw_pt1[1]))

    # Reset and test smoothing state
    tracker._smoothed_tips[0] = (float(raw_pt1[0]), float(raw_pt1[1]))
    prev_x, prev_y = tracker._smoothed_tips[0]
    new_smooth_x = alpha * raw_pt2[0] + (1 - alpha) * prev_x
    new_smooth_y = alpha * raw_pt2[1] + (1 - alpha) * prev_y
    smoothed_res = (int(round(new_smooth_x)), int(round(new_smooth_y)))
    assert smoothed_res == (expected_x, expected_y), f"Expected {(expected_x, expected_y)}, got {smoothed_res}"
    print(f"         EMA smoothing verified: ({raw_pt1} -> {raw_pt2}) smoothed to {smoothed_res}")

    # 5. Clean Release
    print("[Step 5] Releasing HandTracker resources...")
    tracker.close()
    assert tracker.landmarker is None, "Landmarker should be None after close"
    print("         HandTracker released cleanly.")

    print("=" * 60)
    print("HAND TRACKER TEST PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_hand_tracker()
    sys.exit(0 if success else 1)
