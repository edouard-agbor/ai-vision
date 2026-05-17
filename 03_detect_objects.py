"""
03_detect_objects.py - Stage 3 (polished, accurate):
YOLOv8 detection + depth-refined 3D coordinates, physical dimensions,
front-face area, visible object volume.

Visuals: per-class distinct bounding box colors (stable palette),
distance-coded label stripes, semi-transparent panels, HUD,
auto-screenshots for the portfolio.
Measurement: depth-band filter to ignore background through the bbox.
"""

import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat,
    AlignFilter, OBStreamType,
)
from ultralytics import YOLO


# ----------------- Frame helpers -----------------

def frame_to_bgr(frame):
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    raw = frame.get_data()
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(np.ascontiguousarray(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        image = np.resize(np.asanyarray(raw), (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return None


def depth_frame_to_array(frame):
    width = frame.get_width()
    height = frame.get_height()
    raw = frame.get_data()
    raw_bytes = np.ascontiguousarray(raw, dtype=np.uint8)
    return raw_bytes.view(dtype=np.uint16).reshape((height, width))


def colorize_depth(depth_mm):
    clipped = np.clip(depth_mm, 300, 5000)
    norm = ((clipped - 300) / (5000 - 300) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored[depth_mm == 0] = (0, 0, 0)
    return colored


# ----------------- Color palettes -----------------

# Stable, high-contrast BGR colors. Each YOLO class id maps to one
# entry via modulo. Person (0), bottle (39), chair (56), etc. all get
# different colors and stay that color across runs.
CLASS_PALETTE_BGR = [
    (255, 200,   0),   # cyan
    (180, 105, 255),   # hot pink
    ( 80, 230,  80),   # green
    ( 50, 180, 255),   # orange
    (200, 100, 255),   # violet
    (255, 220, 100),   # sky
    (100, 255, 200),   # mint
    (255, 100, 180),   # magenta
    (130, 255, 255),   # bright yellow
    (255, 150,  50),   # azure
    ( 50, 255, 200),   # spring green
    (200, 200, 255),   # peach
]


def color_for_class(class_id):
    return CLASS_PALETTE_BGR[class_id % len(CLASS_PALETTE_BGR)]


def color_for_distance(z_m):
    """Red = close (< 1m), amber = mid (1–2.5m), green = far (> 2.5m).
    Used ONLY for the label panel's left stripe."""
    if z_m < 1.0:
        return (60, 60, 255)
    if z_m < 2.5:
        return (60, 220, 255)
    return (80, 230, 80)


# ----------------- Drawing helpers -----------------

def draw_semi_panel(image, x0, y0, x1, y1, fill=(25, 25, 25), alpha=0.72):
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(image.shape[1], x1)
    y1 = min(image.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    sub = image[y0:y1, x0:x1]
    overlay = np.full_like(sub, fill, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)


def draw_label(image, anchor_x, anchor_y_top, lines, stripe_color, header_zone_h=44):
    """Semi-transparent label panel with distance-coded left stripe."""
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.5
    thick = 1
    pad_x, pad_y = 8, 6
    line_h = 18
    stripe_w = 4

    widths = [cv2.getTextSize(s, font, scale, thick)[0][0] for s in lines]
    panel_w = max(widths) + 2 * pad_x + stripe_w
    panel_h = line_h * len(lines) + 2 * pad_y

    img_h, img_w = image.shape[:2]

    # Above the bbox if there's room; otherwise drop inside the bbox
    # but never overlap the HUD.
    if anchor_y_top - panel_h >= header_zone_h:
        y0 = anchor_y_top - panel_h
    else:
        y0 = max(anchor_y_top + 2, header_zone_h + 4)
    x0 = max(0, min(anchor_x, img_w - panel_w))
    y0 = max(0, min(y0, img_h - panel_h))
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    draw_semi_panel(image, x0, y0, x1, y1, fill=(25, 25, 25), alpha=0.72)
    cv2.rectangle(image, (x0, y0), (x0 + stripe_w, y1), stripe_color, -1)

    for i, line in enumerate(lines):
        ty = y0 + pad_y + line_h * (i + 1) - 5
        cv2.putText(image, line, (x0 + stripe_w + pad_x, ty),
                    font, scale, (240, 240, 240), thick, cv2.LINE_AA)


def draw_center_marker(image, cu, cv_, color):
    cv2.circle(image, (int(cu), int(cv_)), 5, color, -1, cv2.LINE_AA)
    cv2.circle(image, (int(cu), int(cv_)), 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(image, fps, n_detections):
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.6
    thick = 1
    txt = f"FPS  {fps:5.1f}   |   OBJECTS  {n_detections}"
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
    pad = 10
    x0, y0 = 10, 10
    x1, y1 = x0 + tw + 2 * pad, y0 + th + 2 * pad
    draw_semi_panel(image, x0, y0, x1, y1, fill=(15, 15, 15), alpha=0.78)
    cv2.rectangle(image, (x0, y0), (x0 + 4, y1), (0, 230, 120), -1)
    cv2.putText(image, txt, (x0 + 10, y0 + th + pad - 4),
                font, scale, (240, 240, 240), thick, cv2.LINE_AA)


def draw_watermark(image, text="AI VISION  |  STAGE 3"):
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.45
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    img_h, img_w = image.shape[:2]
    x = img_w - tw - 12
    y = img_h - 12
    cv2.putText(image, text, (x + 1, y + 1), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, scale, (210, 210, 210), thick, cv2.LINE_AA)


# ----------------- Measurement core -----------------

def measure_object(roi_depth_mm, fx, fy, cx, cy, x1, y1, x2, y2):
    """
    Given a depth ROI (mm) inside a YOLO bbox, compute the OBJECT's
    3D center + dimensions, rejecting background pixels via a depth band.
    Returns None if too few valid pixels.
    """
    valid = roi_depth_mm[(roi_depth_mm > 300) & (roi_depth_mm < 5000)]
    if valid.size < 50:
        return None

    # Median anchors the object plane
    Z_mm_med = float(np.median(valid))

    # Depth band around the median (±20%). Excludes background wall
    # behind a person, or the table in front of a held object, etc.
    band_frac = 0.20
    lo, hi = Z_mm_med * (1 - band_frac), Z_mm_med * (1 + band_frac)
    object_px = valid[(valid >= lo) & (valid <= hi)]

    # Fallback: widen band if too few survivors
    if object_px.size < 20:
        band_frac = 0.40
        lo, hi = Z_mm_med * (1 - band_frac), Z_mm_med * (1 + band_frac)
        object_px = valid[(valid >= lo) & (valid <= hi)]
        if object_px.size < 10:
            object_px = valid

    Zmin = float(np.percentile(object_px, 5)) / 1000.0
    Zmax = float(np.percentile(object_px, 95)) / 1000.0
    Z = Z_mm_med / 1000.0
    D = max(Zmax - Zmin, 0.01)

    # Refine W and H by finding the tight extent of object-depth pixels
    # within the bbox. This excludes the wall visible around a person's
    # arms, and similar background bleed.
    obj_mask = (roi_depth_mm >= lo) & (roi_depth_mm <= hi) & (roi_depth_mm > 0)
    if obj_mask.sum() >= 30:
        rows_any = np.any(obj_mask, axis=1)
        cols_any = np.any(obj_mask, axis=0)
        if rows_any.any() and cols_any.any():
            r_idx = np.where(rows_any)[0]
            c_idx = np.where(cols_any)[0]
            obj_h_px = float(r_idx[-1] - r_idx[0] + 1)
            obj_w_px = float(c_idx[-1] - c_idx[0] + 1)
        else:
            obj_w_px = float(x2 - x1)
            obj_h_px = float(y2 - y1)
    else:
        obj_w_px = float(x2 - x1)
        obj_h_px = float(y2 - y1)

    W = (obj_w_px / fx) * Z
    H = (obj_h_px / fy) * Z

    uc = (x1 + x2) / 2.0
    vc = (y1 + y2) / 2.0
    X = (uc - cx) * Z / fx
    Y = (vc - cy) * Z / fy

    return {
        'X': X, 'Y': Y, 'Z': Z,
        'W': W, 'H': H, 'D': D,
        'area': W * H, 'volume': W * H * D,
        'center': (uc, vc),
    }


# ----------------- Main -----------------

def main():
    print(">>> Loading YOLOv8n...")
    model = YOLO('yolov8n.pt')
    print(f">>> Model loaded. {len(model.names)} classes available.")

    pipeline = Pipeline()
    config = Config()

    color_profile = (pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                     .get_default_video_stream_profile())
    config.enable_stream(color_profile)
    print(f">>> Color: {color_profile.get_width()}x{color_profile.get_height()} "
          f"@ {color_profile.get_fps()}fps")

    depth_profile = (pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                     .get_default_video_stream_profile())
    config.enable_stream(depth_profile)
    print(f">>> Depth: {depth_profile.get_width()}x{depth_profile.get_height()} "
          f"@ {depth_profile.get_fps()}fps")

    intr = color_profile.get_intrinsic()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.cx, intr.cy
    if fx <= 0 or fy <= 0:
        print(f"!!! Invalid intrinsics fx={fx}, fy={fy}. Aborting.")
        return
    print(f">>> Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    pipeline.start(config)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    print(">>> Pipeline started + AlignFilter ready (depth -> color).")

    cv2.namedWindow("AI Vision - Stage 3: Detection + 3D", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Vision - Stage 3: Detection + 3D", 1280, 480)

    display_w, display_h = 640, 480
    frame_count = 0
    fps = 0.0
    fps_t0 = time.time()

    auto_shots_saved = 0
    last_auto_shot_frame = -10_000
    MAX_AUTO_SHOTS = 3
    AUTO_SHOT_MIN_GAP = 60
    AUTO_SHOT_MIN_CONF = 0.60
    auto_dir = Path("screenshots/auto")

    class_counter = Counter()

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            aligned = align_filter.process(frames)
            if aligned is None:
                continue
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            color_image = frame_to_bgr(color_frame)
            depth_mm = depth_frame_to_array(depth_frame)
            if color_image is None or depth_mm is None:
                continue

            results = model(color_image, verbose=False, conf=0.4, imgsz=320)
            detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]
                    class_counter[label] += 1

                    dh, dw = depth_mm.shape
                    x1c, x2c = max(0, x1), min(dw, x2)
                    y1c, y2c = max(0, y1), min(dh, y2)
                    if x2c <= x1c or y2c <= y1c:
                        continue

                    roi = depth_mm[y1c:y2c, x1c:x2c]
                    meas = measure_object(roi, fx, fy, cx, cy, x1, y1, x2, y2)

                    if meas is None:
                        detections.append({
                            'box': (x1, y1, x2, y2), 'label': label,
                            'cls': cls, 'conf': conf, 'has_3d': False,
                        })
                        continue

                    det = {
                        'box': (x1, y1, x2, y2),
                        'label': label, 'cls': cls, 'conf': conf,
                        'has_3d': True,
                    }
                    det.update(meas)
                    detections.append(det)

            # ---- Draw ----
            depth_colored = colorize_depth(depth_mm)

            for det in detections:
                x1, y1, x2, y2 = det['box']

                if det['has_3d']:
                    cls_color = color_for_class(det['cls'])
                    stripe_color = color_for_distance(det['Z'])
                else:
                    cls_color = (180, 180, 180)
                    stripe_color = (180, 180, 180)

                cv2.rectangle(color_image, (x1, y1), (x2, y2), cls_color, 2, cv2.LINE_AA)
                cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

                if det['has_3d']:
                    lines = [
                        f"{det['label']}  {det['conf']*100:.0f}%",
                        f"{det['W']:.2f} x {det['H']:.2f} x {det['D']:.2f} m  @  {det['Z']:.2f}m",
                        f"A: {det['area']:.3f} m^2   V: {det['volume']:.4f} m^3",
                    ]
                    draw_center_marker(color_image, det['center'][0], det['center'][1], cls_color)
                else:
                    lines = [f"{det['label']}  {det['conf']*100:.0f}%", "no depth"]

                draw_label(color_image, x1, y1, lines, stripe_color)

            draw_hud(color_image, fps, len(detections))
            draw_watermark(color_image)

            depth_title = "DEPTH  (aligned to color)"
            (tw, th), _ = cv2.getTextSize(depth_title, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            draw_semi_panel(depth_colored, 10, 10, 10 + tw + 20, 10 + th + 16,
                            fill=(15, 15, 15), alpha=0.78)
            cv2.putText(depth_colored, depth_title, (20, 10 + th + 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

            color_disp = cv2.resize(color_image, (display_w, display_h))
            depth_disp = cv2.resize(depth_colored, (display_w, display_h))
            divider = np.full((display_h, 2, 3), 20, dtype=np.uint8)
            combined = np.hstack([color_disp, divider, depth_disp])
            cv2.imshow("AI Vision - Stage 3: Detection + 3D", combined)

            if detections:
                mean_conf = float(np.mean([d['conf'] for d in detections]))
                if (auto_shots_saved < MAX_AUTO_SHOTS
                        and mean_conf >= AUTO_SHOT_MIN_CONF
                        and (frame_count - last_auto_shot_frame) >= AUTO_SHOT_MIN_GAP):
                    auto_dir.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    fn = auto_dir / f"auto_{auto_shots_saved+1}_of_3_{stamp}.png"
                    cv2.imwrite(str(fn), combined)
                    auto_shots_saved += 1
                    last_auto_shot_frame = frame_count
                    print(f">>> Auto-screenshot {auto_shots_saved}/{MAX_AUTO_SHOTS} "
                          f"(mean conf {mean_conf*100:.0f}%): {fn}")

            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30.0 / max(now - fps_t0, 1e-6)
                fps_t0 = now
                print(f">>> Frame {frame_count} | FPS: {fps:.1f} | detections: {len(detections)}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                Path("screenshots").mkdir(exist_ok=True)
                fn = f"screenshots/detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fn, combined)
                print(f">>> Manual screenshot saved: {fn}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n>>> Session complete. Total frames: {frame_count}")
        if class_counter:
            print(">>> Detection summary (top 10 classes):")
            for cls_name, cnt in class_counter.most_common(10):
                print(f"      {cls_name:20s}  {cnt} detections")
        print(f">>> Auto-screenshots saved: {auto_shots_saved} in {auto_dir}")


if __name__ == "__main__":
    main()