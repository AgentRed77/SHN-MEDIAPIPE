"""Main Application Entry Point for FogMirror.

Phase 3: Live Hand Landmark Tracking & Index Fingertip Detection
using MediaPipe Tasks API.
"""

import sys
import time
import cv2

from config import CAMERA, TRACKER, WINDOW
from src.camera import Camera
from src.hand_tracker import HandTracker


def run():
    """Executes the main camera and hand tracking loop."""
    print("=" * 60)
    print(" FogMirror - Virtual Fog Glass Writing System ")
    print(" Phase 3: Hand Tracking & Index Fingertip Detection")
    print("=" * 60)

    # Initialize Hardware Camera
    camera = Camera(config=CAMERA)
    if not camera.start():
        print("\n[Error] Unable to initialize webcam stream.")
        return 1

    # Initialize MediaPipe Hand Tracker
    tracker = HandTracker(config=TRACKER)

    # Create resizable OpenCV display window
    cv2.namedWindow(WINDOW.title, cv2.WINDOW_NORMAL)
    w, h = camera.resolution
    if w > 0 and h > 0:
        cv2.resizeWindow(WINDOW.title, w, h)

    print("\n[Info] Controls:")
    print("  - Raise your hand in front of the camera to see tracking.")
    print("  - Landmark 8 (Index Fingertip) is highlighted in yellow/orange.")
    print("  - Press 'q' or 'Q' to quit.")
    print("  - Press 'ESC' to quit.\n")

    # Timing variables for FPS and MediaPipe timestamps
    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                time.sleep(0.005)
                continue

            # Calculate frame timestamp in milliseconds for MediaPipe VIDEO mode
            timestamp_ms = int((time.time() - start_time) * 1000)

            # Process frame for hand landmarks
            tracking_result = tracker.process_frame(frame, timestamp_ms)

            # Draw 21 landmarks and index fingertip highlight
            tracker.draw_landmarks(frame, tracking_result)

            # Calculate smoothed FPS
            curr_time = time.time()
            frame_count += 1
            elapsed = curr_time - prev_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = curr_time

            # Top HUD Banner: Dark translucent bar
            header = frame.copy()
            cv2.rectangle(header, (0, 0), (frame.shape[1], 48), (18, 18, 18), -1)
            cv2.addWeighted(header, 0.75, frame, 0.25, 0, frame)

            # Application title & FPS
            cv2.putText(
                frame,
                "FogMirror [Phase 3: Hand Tracking]",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

            # Hand status badge
            if tracking_result.hand_count == 0:
                status_text = "Hands: 0 (No hand detected)"
                status_color = (130, 130, 130)  # Neutral grey
            elif tracking_result.hand_count == 1:
                tip = tracking_result.primary_index_tip
                norm = tracking_result.primary_index_tip_norm
                if tip and norm:
                    status_text = f"Hands: 1 | Px: ({tip[0]}, {tip[1]}) | Norm: ({norm[0]:.3f}, {norm[1]:.3f})"
                elif tip:
                    status_text = f"Hands: 1 | Px: ({tip[0]}, {tip[1]})"
                else:
                    status_text = "Hands: 1"
                status_color = (80, 220, 80)    # Green
            else:
                status_text = f"Hands: {tracking_result.hand_count} detected"
                status_color = (80, 220, 220)   # Cyan

            cv2.putText(
                frame,
                status_text,
                (450, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
                cv2.LINE_AA,
            )

            # FPS and exit hint
            info_text = f"{fps:.1f} FPS | Press 'Q' to Exit"
            cv2.putText(
                frame,
                info_text,
                (frame.shape[1] - 300, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Display frame
            cv2.imshow(WINDOW.title, frame)

            # Keyboard listener
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("[Info] Exit signal received.")
                break

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
    finally:
        print("[Info] Shutting down cleanly...")
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
        print("[Info] All resources safely released.")

    return 0


if __name__ == "__main__":
    sys.exit(run())
