"""Canvas & Wipe Mask Engine for FogMirror.

Manages the 2D alpha mask that tracks wiped areas on the virtual glass surface.
Provides continuous anti-aliased stroke interpolation, feathered edge generation,
and canvas clearing.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

# Handle relative or package imports
try:
    from config import CanvasConfig, CANVAS
except ImportError:
    from ..config import CanvasConfig, CANVAS


class Canvas:
    """Maintains the fog wiping mask and renders continuous brush strokes.
    
    Attributes:
        width (int): Canvas width in pixels.
        height (int): Canvas height in pixels.
        config (CanvasConfig): Canvas configuration parameters.
    """

    def __init__(self, width: int, height: int, config: Optional[CanvasConfig] = None):
        """Initializes the canvas wipe mask.
        
        Args:
            width: Width of the canvas matching camera frame width.
            height: Height of the canvas matching camera frame height.
            config: Optional CanvasConfig instance. Defaults to CANVAS from config.py.
        """
        self.width = width
        self.height = height
        self.config = config or CANVAS

        # 0 = Completely Fogged, 255 = Completely Cleared
        self._raw_mask = np.zeros((height, width), dtype=np.uint8)

        # Stroke tracking state
        self._prev_point: Optional[Tuple[int, int]] = None
        self._is_drawing: bool = False

        # Cached feathered alpha mask to avoid redundant Gaussian blurs
        self._cached_alpha: Optional[np.ndarray] = None
        self._is_dirty: bool = True

        print(f"[Canvas] Initialized {width}x{height} wipe canvas (Brush Radius: {self.config.brush_radius}px)")

    def start_stroke(self, point: Tuple[int, int]):
        """Begins a new stroke at the given coordinate without connecting to previous strokes.
        
        Args:
            point: (x, y) starting pixel coordinate.
        """
        px, py = self._clamp_point(point)
        self._prev_point = (px, py)
        self._is_drawing = True

        # Draw initial circular brush stamp at stroke start
        cv2.circle(
            self._raw_mask,
            (px, py),
            self.config.brush_radius,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )
        self._is_dirty = True

    def wipe_to(self, point: Tuple[int, int]):
        """Draws a continuous stroke connecting previous point to the new point.
        
        Uses OpenCV anti-aliased lines with round end-caps to ensure rapid finger
        movements never leave disconnected gaps.
        
        Args:
            point: Current (x, y) pixel coordinate of index fingertip.
        """
        px, py = self._clamp_point(point)
        curr_pt = (px, py)

        if not self._is_drawing or self._prev_point is None:
            self.start_stroke(curr_pt)
            return

        prev_pt = self._prev_point

        # Calculate Euclidean distance between consecutive points
        dx = curr_pt[0] - prev_pt[0]
        dy = curr_pt[1] - prev_pt[1]
        distance = np.hypot(dx, dy)

        # Only draw if point moved at least 1 pixel to save cycles
        if distance > 0.5:
            # Draw continuous thick stroke connecting prev and curr points
            cv2.line(
                self._raw_mask,
                prev_pt,
                curr_pt,
                255,
                thickness=self.config.brush_radius * 2,
                lineType=cv2.LINE_AA,
            )
            # Add round end cap to smooth stroke joint
            cv2.circle(
                self._raw_mask,
                curr_pt,
                self.config.brush_radius,
                255,
                -1,
                lineType=cv2.LINE_AA,
            )
            self._is_dirty = True

        self._prev_point = curr_pt

    def end_stroke(self):
        """Ends the active stroke sequence, resetting interpolation history."""
        self._prev_point = None
        self._is_drawing = False

    def clear(self):
        """Resets the canvas back to full fog (all zeros)."""
        self._raw_mask.fill(0)
        self._prev_point = None
        self._is_drawing = False
        self._is_dirty = True
        print("[Canvas] Surface cleared. Fog fully restored.")

    def get_mask(self) -> np.ndarray:
        """Returns the raw uint8 mask array (0 to 255)."""
        return self._raw_mask

    def get_alpha_mask(self) -> np.ndarray:
        """Returns a normalized, feathered float32 alpha mask in range [0.0, 1.0].
        
        Applies a Gaussian blur to create soft, natural boundaries along wiped edges.
        Caches the result if the mask was not modified since the last call.
        
        Returns:
            np.ndarray: 2D float32 array where 0.0 = full fog and 1.0 = full clear.
        """
        if not self._is_dirty and self._cached_alpha is not None:
            return self._cached_alpha

        feather_ksize = self.config.feather_radius
        # Ensure kernel size is odd and >= 3
        if feather_ksize % 2 == 0:
            feather_ksize += 1
        feather_ksize = max(3, feather_ksize)

        # Apply soft edge feathering
        feathered = cv2.GaussianBlur(self._raw_mask, (feather_ksize, feather_ksize), 0)

        # Normalize to [0.0, 1.0] float32 for vectorized alpha blending
        self._cached_alpha = feathered.astype(np.float32) / 255.0
        self._is_dirty = False

        return self._cached_alpha

    @property
    def is_empty(self) -> bool:
        """Returns True if no fog has been cleared yet."""
        return not np.any(self._raw_mask > 0)

    @property
    def cleared_percentage(self) -> float:
        """Returns the percentage of surface currently wiped clear (0.0 to 100.0)."""
        total_pixels = self.width * self.height
        if total_pixels == 0:
            return 0.0
        cleared_pixels = np.count_nonzero(self._raw_mask > 128)
        return (cleared_pixels / total_pixels) * 100.0

    def _clamp_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Clamps (x, y) coordinates within canvas bounds."""
        x = int(np.clip(point[0], 0, self.width - 1))
        y = int(np.clip(point[1], 0, self.height - 1))
        return x, y
