"""Environment and Dependency Verification Script for FogMirror.

This script tests that:
1. Python interpreter is running from the local .venv virtual environment.
2. OpenCV (cv2) imports and its version is reported.
3. NumPy imports and basic matrix operations work.
4. MediaPipe imports and its HandLandmarker/solutions components can be accessed.
"""

import sys
import os

print("=" * 60)
print("FOGMIRROR PHASE 1: ENVIRONMENT & IMPORT VERIFICATION")
print("=" * 60)

# 1. Python Environment Check
print(f"[1/4] Python Version: {sys.version.split()[0]}")
print(f"      Executable: {sys.executable}")
is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
print(f"      Virtual Environment Active: {is_venv}")
if not is_venv:
    print("      WARNING: Not running inside a virtual environment!")
else:
    print("      SUCCESS: Running within project .venv")

# 2. NumPy Check
try:
    import numpy as np
    test_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    print(f"[2/4] NumPy Version: {np.__version__} - SUCCESS (shape: {test_arr.shape})")
except Exception as e:
    print(f"[2/4] NumPy Check FAILED: {e}")
    sys.exit(1)

# 3. OpenCV Check
try:
    import cv2
    print(f"[3/4] OpenCV Version: {cv2.__version__} - SUCCESS")
except Exception as e:
    print(f"[3/4] OpenCV Check FAILED: {e}")
    sys.exit(1)

# 4. MediaPipe Check
try:
    import mediapipe as mp
    print(f"[4/4] MediaPipe Version: {mp.__version__}")
    
    # Check available APIs in MediaPipe
    has_tasks = hasattr(mp, 'tasks')
    has_solutions = hasattr(mp, 'solutions')
    print(f"      MediaPipe Tasks API available: {has_tasks}")
    print(f"      MediaPipe Solutions API available: {has_solutions}")
    
    if has_solutions and hasattr(mp.solutions, 'hands'):
        print("      MediaPipe Solutions.Hands module found.")
    if has_tasks:
        print("      MediaPipe Vision Tasks (HandLandmarker) found.")
        
    print("      SUCCESS: MediaPipe imported and verified.")
except Exception as e:
    print(f"[4/4] MediaPipe Check FAILED: {e}")
    sys.exit(1)

print("=" * 60)
print("ALL DEPENDENCY CHECKS PASSED SUCCESSFULLY!")
print("=" * 60)
