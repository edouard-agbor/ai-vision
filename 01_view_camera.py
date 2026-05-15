"""
01_view_camera.py - Debug version
Shows live video AND prints frame-by-frame status so we can see what's happening.
"""

import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat


def frame_to_bgr(frame):
    """Convert Orbbec colour frame to BGR for OpenCV display."""
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    raw = frame.get_data()

    # MJPG is JPEG-compressed bytes — decode with explicit uint8 buffer.
    if fmt == OBFormat.MJPG:
        data = np.ascontiguousarray(raw, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    data = np.asanyarray(raw)

    if fmt == OBFormat.RGB:
        image = data.reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape((height, width, 3))
    if fmt == OBFormat.NV12:
        yuv = data.reshape((int(height * 1.5), width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.YUYV:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    return None


def main():
    pipeline = Pipeline()
    config = Config()

    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    pipeline.start(config)
    print(">>> Camera started, waiting for first frame...")

    # Force window to be created BEFORE the loop. This helps Windows recognise it.
    cv2.namedWindow("AI Vision - Color Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Vision - Color Stream", 800, 600)

    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                print(">>> wait_for_frames returned None")
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                print(">>> get_color_frame returned None")
                continue

            # Print details on first frame so we see what's coming in.
            if frame_count == 0:
                print(f">>> First frame received!")
                print(f"    Format: {color_frame.get_format()}")
                print(f"    Size: {color_frame.get_width()}x{color_frame.get_height()}")

            image = frame_to_bgr(color_frame)

            if image is None:
                if frame_count % 15 == 0:
                    print(f">>> Frame {frame_count}: conversion to BGR failed")
                frame_count += 1
                continue

            if frame_count == 0:
                print(f"    Converted image shape: {image.shape}")
                print(">>> If you don't see a window now, check taskbar or Alt+Tab")

            cv2.imshow("AI Vision - Color Stream", image)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f">>> {frame_count} frames displayed")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f">>> Camera stopped. Total frames: {frame_count}")


if __name__ == "__main__":
    main()