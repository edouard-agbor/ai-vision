"""
02_view_depth.py - Stage 2: Color + Depth side-by-side with live distance measurement
"""

import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
from datetime import datetime
from pathlib import Path


def frame_to_bgr(frame):
    """Convert Orbbec colour frame to BGR for OpenCV display."""
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    raw = frame.get_data()

    if fmt == OBFormat.MJPG:
        data = np.ascontiguousarray(raw, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        image = np.resize(np.asanyarray(raw), (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return None


def depth_to_colorized(frame):
    """Convert Orbbec depth frame to a colorized heat map + return raw depth in mm."""
    width = frame.get_width()
    height = frame.get_height()
    raw = frame.get_data()

    # Depth comes as raw bytes — reinterpret (view) as uint16, not cast.
    # Each depth pixel is 2 bytes representing distance in millimeters.
    raw_bytes = np.ascontiguousarray(raw, dtype=np.uint8)
    depth_mm = raw_bytes.view(dtype=np.uint16).reshape((height, width))

    # Clip to reasonable indoor range: 30cm to 5m
    depth_clipped = np.clip(depth_mm, 300, 5000)

    # Normalize to 0-255 for color mapping (closer = lower value)
    depth_normalized = ((depth_clipped - 300) / (5000 - 300) * 255).astype(np.uint8)

    # Apply JET colormap: close = red/yellow, far = blue
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # Mask out invalid depth (where sensor saw nothing) as black
    depth_colored[depth_mm == 0] = (0, 0, 0)

    return depth_colored, depth_mm


def main():
    pipeline = Pipeline()
    config = Config()

    # Enable color stream
    color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)
    print(f">>> Color: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps")

    # Enable depth stream
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
    print(f">>> Depth: {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()}fps")

    pipeline.start(config)
    print(">>> Pipeline started. Streaming color + depth...")

    cv2.namedWindow("AI Vision - Stage 2: Color + Depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Vision - Stage 2: Color + Depth", 1280, 480)

    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            color_image = frame_to_bgr(color_frame)
            depth_image, depth_mm = depth_to_colorized(depth_frame)
            if color_image is None or depth_image is None:
                continue

            # Resize both to a common size for side-by-side display
            display_w, display_h = 640, 480
            color_resized = cv2.resize(color_image, (display_w, display_h))
            depth_resized = cv2.resize(depth_image, (display_w, display_h))

            # Sample distance at the CENTER of the depth frame
            cy, cx = depth_mm.shape[0] // 2, depth_mm.shape[1] // 2
            # Average a small 5x5 patch around center for stability (some pixels may be 0)
            patch = depth_mm[cy-2:cy+3, cx-2:cx+3]
            valid = patch[patch > 0]
            distance_mm = int(valid.mean()) if valid.size > 0 else 0
            distance_m = distance_mm / 1000.0

            # Draw crosshair + distance label on BOTH images
            for img in (color_resized, depth_resized):
                cx_d, cy_d = display_w // 2, display_h // 2
                cv2.line(img, (cx_d - 20, cy_d), (cx_d + 20, cy_d), (0, 255, 0), 2)
                cv2.line(img, (cx_d, cy_d - 20), (cx_d, cy_d + 20), (0, 255, 0), 2)
                label = f"{distance_m:.2f} m" if distance_mm > 0 else "no data"
                cv2.putText(img, label, (cx_d + 25, cy_d - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Add stream labels
            cv2.putText(color_resized, "COLOR", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(depth_resized, "DEPTH (red=close, blue=far)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Stack side by side
            combined = np.hstack([color_resized, depth_resized])
            cv2.imshow("AI Vision - Stage 2: Color + Depth", combined)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f">>> Frame {frame_count} | Center distance: {distance_m:.2f}m")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                Path("screenshots").mkdir(exist_ok=True)
                filename = f"screenshots/depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, combined)
                print(f">>> Saved: {filename}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f">>> Stopped. Total frames: {frame_count}")


if __name__ == "__main__":
    main()