"""Hand Tracking Module for FogMirror.

Integrates MediaPipe's modern Tasks API (HandLandmarker) to detect hands,
extract all 21 3D landmarks, apply temporal smoothing to the index fingertip,
and render visual debugging overlays.
"""

import os
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Handle relative or package imports
try:
    from config import TrackerConfig, TRACKER
except ImportError:
    from ..config import TrackerConfig, TRACKER


# Standard 21 Hand Landmark Connections for Skeleton Rendering
HAND_CONNECTIONS = [
    # Palm base & knuckles
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index Finger
    (5, 6), (6, 7), (7, 8),
    # Middle Finger
    (9, 10), (10, 11), (11, 12),
    # Ring Finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
]


@dataclass
class HandData:
    """Contains tracked information for a single detected hand.
    
    Attributes:
        landmarks: List of 21 (x, y) tuples in pixel coordinates.
        landmarks_norm: List of 21 raw normalized landmarks (x, y, z in [0, 1]).
        handedness: Hand label ("Left" or "Right").
        score: Detection confidence score (0.0 to 1.0).
        index_tip: Raw (x, y) pixel coordinate of landmark 8.
        index_tip_smoothed: Filtered (x, y) pixel coordinate after EMA smoothing.
        index_tip_norm: Raw normalized (x, y, z) coordinates of landmark 8 in [0, 1].
    """
    landmarks: List[Tuple[int, int]]
    landmarks_norm: List[Tuple[float, float, float]]
    handedness: str
    score: float
    index_tip: Tuple[int, int]
    index_tip_smoothed: Tuple[int, int]
    index_tip_norm: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class TrackingResult:
    """Container for the output of a single video frame analysis."""
    hands: List[HandData] = field(default_factory=list)

    @property
    def hand_count(self) -> int:
        return len(self.hands)

    @property
    def primary_index_tip(self) -> Optional[Tuple[int, int]]:
        """Returns the smoothed index fingertip of the first detected hand, if any."""
        if self.hands:
            return self.hands[0].index_tip_smoothed
        return None

    @property
    def primary_index_tip_norm(self) -> Optional[Tuple[float, float, float]]:
        """Returns normalized (x, y, z) of the first detected hand's index tip."""
        if self.hands:
            return self.hands[0].index_tip_norm
        return None


class HandTracker:
    """Detects hand landmarks using MediaPipe HandLandmarker Tasks API."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        """Initializes HandLandmarker and ensures model asset exists.
        
        Args:
            config: Optional TrackerConfig. Defaults to TRACKER from config.py.
        """
        self.config = config or TRACKER
        self._ensure_model_exists()

        # Initialize MediaPipe HandLandmarker in VIDEO running mode
        base_options = python.BaseOptions(model_asset_path=self.config.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # Exponential Moving Average state for fingertip smoothing: hand_idx -> (x, y)
        self._smoothed_tips: dict[int, Tuple[float, float]] = {}

        print(f"[HandTracker] Initialized MediaPipe HandLandmarker (Max Hands: {self.config.num_hands})")

    def _ensure_model_exists(self):
        """Verifies model file exists locally; downloads if missing."""
        if not os.path.exists(self.config.model_path):
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            print(f"[HandTracker] Downloading model asset from {self.config.model_url}...")
            urllib.request.urlretrieve(self.config.model_url, self.config.model_path)
            print(f"[HandTracker] Model saved to {self.config.model_path}")

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> TrackingResult:
        """Processes a single BGR video frame and extracts 21 landmarks per hand.
        
        Args:
            frame_bgr: OpenCV BGR frame (numpy uint8 array).
            timestamp_ms: Monotonically increasing frame timestamp in milliseconds.
            
        Returns:
            TrackingResult containing detected HandData instances.
        """
        h, w, _ = frame_bgr.shape

        # Convert OpenCV BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Synchronous inference in VIDEO mode
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        tracking_result = TrackingResult()

        if not result or not result.hand_landmarks:
            # If no hands detected, clear smoothing memory to prevent jumpy re-entry
            self._smoothed_tips.clear()
            return tracking_result

        alpha = self.config.smoothing_alpha

        for hand_idx, landmarks in enumerate(result.hand_landmarks):
            pixel_landmarks: List[Tuple[int, int]] = []
            norm_landmarks: List[Tuple[float, float, float]] = []

            for lm in landmarks:
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                pixel_landmarks.append((px, py))
                norm_landmarks.append((lm.x, lm.y, lm.z))

            # Handedness label and confidence score
            handedness = "Hand"
            score = 1.0
            if result.handedness and hand_idx < len(result.handedness) and result.handedness[hand_idx]:
                cat = result.handedness[hand_idx][0]
                handedness = cat.category_name
                score = cat.score

            # Landmark 8: Index Fingertip
            raw_tip = pixel_landmarks[8]
            raw_tip_norm = norm_landmarks[8]

            # Apply Exponential Moving Average (EMA) smoothing
            if hand_idx in self._smoothed_tips:
                prev_x, prev_y = self._smoothed_tips[hand_idx]
                smooth_x = alpha * raw_tip[0] + (1.0 - alpha) * prev_x
                smooth_y = alpha * raw_tip[1] + (1.0 - alpha) * prev_y
            else:
                smooth_x, smooth_y = float(raw_tip[0]), float(raw_tip[1])

            self._smoothed_tips[hand_idx] = (smooth_x, smooth_y)
            smoothed_tip = (int(round(smooth_x)), int(round(smooth_y)))

            hand_data = HandData(
                landmarks=pixel_landmarks,
                landmarks_norm=norm_landmarks,
                handedness=handedness,
                score=score,
                index_tip=raw_tip,
                index_tip_smoothed=smoothed_tip,
                index_tip_norm=raw_tip_norm,
            )
            tracking_result.hands.append(hand_data)

        return tracking_result

    def draw_landmarks(self, frame: np.ndarray, tracking_result: TrackingResult) -> np.ndarray:
        """Renders hand skeleton connections and highlights Landmark 8 (index fingertip).
        
        Args:
            frame: Video frame to draw on (modified in place).
            tracking_result: Output from process_frame.
            
        Returns:
            The annotated frame.
        """
        for hand in tracking_result.hands:
            # 1. Draw Skeleton Connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                pt1 = hand.landmarks[start_idx]
                pt2 = hand.landmarks[end_idx]
                cv2.line(frame, pt1, pt2, (180, 220, 100), 2, cv2.LINE_AA)

            # 2. Draw Landmark Nodes
            for idx, pt in enumerate(hand.landmarks):
                if idx == 8:
                    continue  # Index tip is specially drawn below
                # Small neat circular nodes
                cv2.circle(frame, pt, 4, (50, 180, 240), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Specially Highlight Landmark 8 (Index Fingertip)
            tip_x, tip_y = hand.index_tip_smoothed

            # Outer glowing halo ring
            cv2.circle(frame, (tip_x, tip_y), 15, (0, 255, 255), 2, cv2.LINE_AA)
            # Inner solid fingertip point
            cv2.circle(frame, (tip_x, tip_y), 7, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (tip_x, tip_y), 3, (255, 255, 255), -1, cv2.LINE_AA)

            # 4. Display Fingertip Coordinates Tag (Both Pixel & Normalized)
            norm_x, norm_y, _ = hand.index_tip_norm
            line1 = f"Tip Px: ({tip_x}, {tip_y})"
            line2 = f"Norm: ({norm_x:.3f}, {norm_y:.3f})"
            badge_x = tip_x + 18
            badge_y = tip_y - 18

            # High-contrast dual-line badge background
            cv2.rectangle(
                frame,
                (badge_x - 4, badge_y - 14),
                (badge_x + 160, badge_y + 24),
                (20, 20, 20),
                -1,
            )
            cv2.putText(
                frame,
                line1,
                (badge_x, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line2,
                (badge_x, badge_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (160, 240, 160),
                1,
                cv2.LINE_AA,
            )

        return frame

    def close(self):
        """Releases the MediaPipe Landmarker."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()
            self.landmarker = None
            print("[HandTracker] MediaPipe Landmarker released cleanly.")
