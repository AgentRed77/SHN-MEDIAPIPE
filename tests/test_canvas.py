"""Automated unit test for Canvas & Wipe Mask Engine.

Tests:
1. Canvas initialization & empty state.
2. Stroke interpolation & continuity (no gaps).
3. Stroke termination (no unwanted bridge lines).
4. Feathered alpha mask generation & normalization.
5. Canvas clearing & reset.
6. Execution latency benchmark.
"""

import sys
import os
import time
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CanvasConfig
from src.canvas import Canvas


def test_canvas_engine():
    print("=" * 60)
    print("RUNNING CANVAS ENGINE UNIT & BENCHMARK TEST")
    print("=" * 60)

    w, h = 1280, 720
    config = CanvasConfig(brush_radius=20, feather_radius=11)
    canvas = Canvas(w, h, config=config)

    # 1. Initial State Check
    print("[Step 1] Checking initial empty canvas state...")
    assert canvas.is_empty, "Canvas must be empty at initialization"
    assert canvas.cleared_percentage == 0.0, "Cleared percentage should be 0.0"
    mask = canvas.get_mask()
    assert mask.shape == (h, w), f"Mask shape must be ({h}, {w})"
    assert np.all(mask == 0), "All mask pixels should initially be 0 (full fog)"
    print("         Initial state verified.")

    # 2. Continuous Stroke Interpolation Test
    print("[Step 2] Testing continuous stroke drawing & interpolation...")
    pt1 = (200, 300)
    pt2 = (400, 300)  # Horizontal line across 200 pixels
    canvas.start_stroke(pt1)
    canvas.wipe_to(pt2)

    # Verify start, end, and midpoint are all cleared (255)
    mid_pt = (300, 300)
    raw_mask = canvas.get_mask()
    assert raw_mask[pt1[1], pt1[0]] == 255, "Stroke start must be cleared"
    assert raw_mask[mid_pt[1], mid_pt[0]] == 255, "Stroke midpoint must be continuously cleared (no gaps)"
    assert raw_mask[pt2[1], pt2[0]] == 255, "Stroke end must be cleared"
    assert not canvas.is_empty, "Canvas should not be empty after drawing"
    assert canvas.cleared_percentage > 0.0, "Cleared percentage should be positive"
    print(f"         Stroke continuity verified: {canvas.cleared_percentage:.2f}% cleared.")

    # 3. Discontinuous Stroke Separation Test
    print("[Step 3] Testing stroke termination (no connecting bridge lines)...")
    canvas.end_stroke()
    pt3 = (600, 300)  # Jump 200px to the right
    canvas.start_stroke(pt3)

    # The region between pt2 (400, 300) and pt3 (600, 300) should remain fogged (0)
    gap_midpoint = (500, 300)
    raw_mask = canvas.get_mask()
    assert raw_mask[gap_midpoint[1], gap_midpoint[0]] == 0, "No bridge line should connect separated strokes"
    print("         Stroke separation verified (gap remains fogged).")

    # 4. Feathered Alpha Mask Test
    print("[Step 4] Testing feathered alpha mask generation...")
    alpha = canvas.get_alpha_mask()
    assert alpha.shape == (h, w), "Alpha mask shape must match canvas dimensions"
    assert alpha.dtype == np.float32, "Alpha mask dtype must be float32"
    assert np.min(alpha) >= 0.0 and np.max(alpha) <= 1.0, "Alpha values must be bounded in [0.0, 1.0]"

    # Verify gradient/feathered edge exists (values between 0.1 and 0.9)
    has_gradient = np.any((alpha > 0.1) & (alpha < 0.9))
    assert has_gradient, "Feathered edge must produce intermediate gradient values"
    print("         Feathered alpha mask verified (gradient present, strictly [0.0, 1.0]).")

    # 5. Canvas Clear Test
    print("[Step 5] Testing canvas clear...")
    canvas.clear()
    assert canvas.is_empty, "Canvas must be empty after clear()"
    assert canvas.cleared_percentage == 0.0, "Cleared percentage must reset to 0.0"
    alpha_cleared = canvas.get_alpha_mask()
    assert np.all(alpha_cleared == 0.0), "Alpha mask must be all 0.0 after clear()"
    print("         Canvas reset verified.")

    # 6. Performance & Latency Benchmark
    print("[Step 6] Benchmarking stroke & alpha mask generation latency (100 iterations)...")
    t0 = time.perf_counter()
    canvas.start_stroke((100, 100))
    for i in range(100):
        canvas.wipe_to((100 + i * 5, 100 + i * 3))
        _ = canvas.get_alpha_mask()
    canvas.end_stroke()
    total_ms = (time.perf_counter() - t0) * 1000
    avg_ms = total_ms / 100
    print(f"         Total time for 100 stroke+feather operations: {total_ms:.1f} ms")
    print(f"         Average per-frame latency: {avg_ms:.3f} ms")
    assert avg_ms < 3.0, f"Canvas latency must be under 3.0 ms, got {avg_ms:.3f} ms"

    print("=" * 60)
    print("CANVAS ENGINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_canvas_engine()
    sys.exit(0 if success else 1)
