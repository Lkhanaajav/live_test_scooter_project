# camera_waypoint_pipeline.py
# Adapted from modular_waypoint_pipeline.py for real-time camera input.
# Keeps the same logic for road detection and path planning.

import os
import cv2
import math
import numpy as np
import networkx as nx
import sys
import time
import argparse

# ---- Your detector ----
from fast_road_detector import FastRoadDetector, Config

# =============================================================================
# Configuration (hardcoded for minimalism; adjust as needed)
# =============================================================================
ROAD_ID = 1
SIDEWALK_ID = 2

src_points = np.array([
    [0.0,   717.0],
    [1278.0, 717.0],
    [860.0,  337.0],
    [573.0,  329.0]
], dtype=np.float32)

dst_points = np.array([
    [100, 480],  # bottom-left
    [500, 480],  # bottom-right
    [400, 100],  # top-right
    [200, 100]   # top-left
], dtype=np.float32)

bev_size = (600, 500)  # (W, H)
H = cv2.getPerspectiveTransform(src_points, dst_points)
Hinv = np.linalg.inv(H)

TRIM_BOTTOM = 20

PATH_COLORS = [
    (0,255,255), (255,255,0), (255,0,255),
    (0,165,255), (0,255,128), (128,0,255),
    (255,128,0), (0,128,255), (128,255,0)
]

# Default camera settings (adjust here)
DEFAULT_CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
DISPLAY_HEIGHT = 300  # For real-time view

# =============================================================================
# Utilities (copied from modular_waypoint_pipeline.py)
# =============================================================================
def split_masks_from_output(model_output, road_id=ROAD_ID, sidewalk_id=SIDEWALK_ID):
    m = model_output.astype(np.uint8) if model_output.dtype != np.uint8 else model_output
    uniq = set(np.unique(m).tolist())
    if uniq.issubset({0, 255}):
        sidewalk = (m > 0).astype(np.uint8) * 255
        road = np.zeros_like(sidewalk, dtype=np.uint8)
        return sidewalk, road
    sidewalk = (m == sidewalk_id).astype(np.uint8) * 255
    road     = (m == road_id).astype(np.uint8) * 255
    return sidewalk, road

def colorize_sidewalk_road(frame_bgr, sidewalk_mask_255, road_mask_255, alpha=0.45):
    overlay = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)
    color_layer[road_mask_255 > 0]     = (255, 120, 0)
    color_layer[sidewalk_mask_255 > 0] = (0, 200, 0)
    cv2.addWeighted(color_layer, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def draw_ribbon(img, pts, color, width=14, glow=6):
    h, w = img.shape[:2]
    pts = [(x, y) for (x, y) in pts if 0 <= x < w and 0 <= y < h]
    if len(pts) < 2:
        return img
    layer = np.zeros_like(img)
    cv2.polylines(layer, [np.int32(pts)], False, color, width, cv2.LINE_AA)
    if glow > 0:
        blur = cv2.GaussianBlur(layer, (0,0), glow)
        img = cv2.addWeighted(img, 1.0, blur, 0.25, 0)
    return cv2.add(img, layer)

# =============================================================================
# GPS utilities (NMEA over serial)
# =============================================================================
def _nmea_to_decimal(raw, hemi):
    if not raw:
        return None
    try:
        raw = float(raw)
    except ValueError:
        return None
    deg = int(raw // 100)
    minutes = raw - deg * 100
    val = deg + minutes / 60.0
    if hemi in ("S", "W"):
        val = -val
    return val

def parse_nmea_latlon(sentence):
    parts = sentence.split(",")
    if len(parts) < 6:
        return None, None
    if parts[0].endswith("GGA"):
        lat = _nmea_to_decimal(parts[2], parts[3])
        lon = _nmea_to_decimal(parts[4], parts[5])
        return lat, lon
    if parts[0].endswith("RMC"):
        lat = _nmea_to_decimal(parts[3], parts[4])
        lon = _nmea_to_decimal(parts[5], parts[6])
        return lat, lon
    return None, None

def gps_test(device, baud=9600):
    try:
        import serial
    except Exception as exc:
        print("❌ pyserial not installed. Run: pip install pyserial")
        raise exc

    print(f"📡 Reading GPS from {device} @ {baud} baud (CTRL+C to stop)")
    with serial.Serial(device, baud, timeout=1) as ser:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line.startswith("$"):
                continue
            lat, lon = parse_nmea_latlon(line)
            if lat is not None and lon is not None:
                print(f"GPS: {lat:.6f}, {lon:.6f} | {line}")

# =============================================================================
# Skeletonization (Zhang–Suen) 
# =============================================================================
def zhang_suen_thinning(bin_img_0_255):
    img = (bin_img_0_255 > 0).astype(np.uint8)
    h, w = img.shape

    def neighbors(y, x):
        return [img[y-1, x], img[y-1, x+1], img[y, x+1], img[y+1, x+1],
                img[y+1, x], img[y+1, x-1], img[y, x-1], img[y-1, x-1]]

    def transitions(nb):
        return sum((nb[i] == 0 and nb[(i+1) % 8] == 1) for i in range(8))

    while True:
        changing1 = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if img[y, x] != 1: continue
                nb = neighbors(y, x); C = transitions(nb); N = sum(nb)
                if (2 <= N <= 6 and C == 1 and nb[0]*nb[2]*nb[4] == 0 and nb[2]*nb[4]*nb[6] == 0):
                    changing1.append((y, x))
        for y, x in changing1: img[y, x] = 0

        changing2 = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if img[y, x] != 1: continue
                nb = neighbors(y, x); C = transitions(nb); N = sum(nb)
                if (2 <= N <= 6 and C == 1 and nb[0]*nb[2]*nb[6] == 0 and nb[0]*nb[4]*nb[6] == 0):
                    changing2.append((y, x))
        for y, x in changing2: img[y, x] = 0

        if not changing1 and not changing2: break

    return (img * 255).astype(np.uint8)

# =============================================================================
# BEV skeleton & graph - copied
# =============================================================================
def extract_skeleton_graph(bev_binary_0_255, trim_px=5):
    kernel = np.ones((5, 5), np.uint8)
    bev_clean = cv2.morphologyEx(bev_binary_0_255, cv2.MORPH_CLOSE, kernel)
    bev_clean = cv2.medianBlur(bev_clean, 5)
    _, binary = cv2.threshold(bev_clean, 127, 255, cv2.THRESH_BINARY)

    skeleton = zhang_suen_thinning(binary)
    sk = skeleton.copy()

    if trim_px > 0:
        sk[:trim_px, :]  = 0
        sk[:, :trim_px]  = 0
        sk[:, -trim_px:] = 0
        sk[-trim_px:, :] = 0

    G = nx.Graph()
    h, w = sk.shape
    for y in range(h):
        xs = np.where(sk[y] == 255)[0]
        for x in xs:
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dx == 0 and dy == 0: continue
                    ny, nx_ = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx_ < w and sk[ny,nx_] == 255:
                        G.add_edge((x,y),(nx_,ny),weight=math.hypot(dx,dy))
    return sk, G

def skeleton_endpoints(G):
    return [n for n in G.nodes if G.degree[n] == 1]

def project_points_bev_to_cam(points_bev):
    if not points_bev: return []
    pts = np.array(points_bev, np.float32).reshape(-1,1,2)
    cam = cv2.perspectiveTransform(pts, Hinv).reshape(-1,2)
    return [(float(x),float(y)) for x,y in cam]

# =============================================================================
# Model init - 
# =============================================================================
def initialize_model():
    cfg = Config(model_dir="models/my-segformer-road_new", conf_thresh=0.5, road_id=ROAD_ID)
    return FastRoadDetector(cfg)

# =============================================================================
# Main (adapted for camera)
# =============================================================================
def process_camera(
    output_dir,
    stride=1,
    save_video=False,
    camera_id=DEFAULT_CAMERA_ID,
    resize_w=None,
    resize_h=None,
    show_window=True,
    fps_log_interval=2.0
):
    print("🔧 Initializing FastRoadDetector...")
    model = initialize_model(); print("✅ Model ready!")

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📹 Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open camera {camera_id}")
        print("Check: Camera connected? Try different ID (0,1,2).")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    vw = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = CAMERA_FPS
        w = CAMERA_WIDTH
        h = CAMERA_HEIGHT
        vw = cv2.VideoWriter(os.path.join(output_dir,"cam_paths.mp4"),fourcc,fps,(w,h))

    frame_id = 0
    fps_count = 0
    fps_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera frame lost, retrying...")
            continue
        if frame_id % stride != 0:
            frame_id += 1
            continue

        infer_frame = frame
        if resize_w and resize_h:
            infer_frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

        model_out, _ = model.process_frame(infer_frame)
        if model_out.shape != infer_frame.shape[:2]:
            model_out = cv2.resize(model_out, (infer_frame.shape[1], infer_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        sidewalk_mask, road_mask = split_masks_from_output(model_out)

        if infer_frame.shape[:2] != frame.shape[:2]:
            sidewalk_mask = cv2.resize(sidewalk_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            road_mask = cv2.resize(road_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        cam_overlay = colorize_sidewalk_road(frame, sidewalk_mask, road_mask)

        bev_sidewalk = cv2.warpPerspective(sidewalk_mask, H, bev_size)
        bev_road     = cv2.warpPerspective(road_mask, H, bev_size)

        if TRIM_BOTTOM > 0:
            bev_sidewalk = bev_sidewalk[:bev_sidewalk.shape[0]-TRIM_BOTTOM, :]
            bev_road     = bev_road[:bev_road.shape[0]-TRIM_BOTTOM, :]

        skeleton_mask, graph = extract_skeleton_graph(bev_sidewalk, trim_px=5)
        endpoints = skeleton_endpoints(graph)

        start = None
        if endpoints:
            start = max(endpoints, key=lambda p: p[1])

        H_bev,W_bev = skeleton_mask.shape
        bev_color = np.zeros((H_bev,W_bev,3),dtype=np.uint8)
        bev_color[bev_road>0]=(255,120,0)
        bev_color[bev_sidewalk>0]=(0,200,0)
        gy,gx=np.where(skeleton_mask>0)
        bev_color[gy,gx]=(0,128,0)

        cam_paths = cam_overlay.copy()

        if start:
            other_endpoints = [e for e in endpoints if e != start]
            for idx, end in enumerate(other_endpoints):
                try:
                    path = nx.dijkstra_path(graph, start, end, weight="weight")
                except nx.NetworkXNoPath:
                    continue
                color = PATH_COLORS[idx % len(PATH_COLORS)]
                # draw on BEV
                for i in range(len(path)-1):
                    x1,y1 = path[i]; x2,y2 = path[i+1]
                    cv2.line(bev_color, (int(x1),int(y1)), (int(x2),int(y2)), color, 2, cv2.LINE_AA)
                cv2.circle(bev_color,(int(start[0]),int(start[1])),6,(0,0,255),-1)
                cv2.circle(bev_color,(int(end[0]),int(end[1])),4,color,-1)

                # draw on cam
                cam_pts = project_points_bev_to_cam(path)
                cam_paths = draw_ribbon(cam_paths, cam_pts, color=color, width=18, glow=6)

        # Save (optional)
        if save_video:
            cv2.imwrite(os.path.join(output_dir,f"bev_paths_{frame_id:04d}.png"),bev_color)
            cv2.imwrite(os.path.join(output_dir,f"cam_paths_{frame_id:04d}.png"),cam_paths)
        if vw: vw.write(cam_paths)

        # Real-time display (optional)
        if show_window:
            bev_display = cv2.resize(bev_color, (int(DISPLAY_HEIGHT * bev_color.shape[1] / bev_color.shape[0]), DISPLAY_HEIGHT))
            cam_display = cv2.resize(cam_paths, (int(DISPLAY_HEIGHT * cam_paths.shape[1] / cam_paths.shape[0]), DISPLAY_HEIGHT))
            combined = np.hstack((cam_display, bev_display))

            cv2.imshow("Real-time Path Planning (Camera | BEV)", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                print("Stopping camera processing...")
                break

        fps_count += 1
        now = time.time()
        if now - fps_start >= fps_log_interval:
            fps = fps_count / (now - fps_start)
            print(f"FPS: {fps:.2f} | frame={frame_id} | start={start} | num_paths={len(other_endpoints) if start else 0}")
            fps_start = now
            fps_count = 0
        frame_id += 1

    cap.release()
    if vw: vw.release()
    cv2.destroyAllWindows()
    print("✅ Done. Any saved results in", output_dir)

# =============================================================================
# Entry
# =============================================================================
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Camera pipeline with FPS logging and resizing")
    parser.add_argument("--camera-id", type=int, default=DEFAULT_CAMERA_ID, help="Camera device ID")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--resize-w", type=int, default=None, help="Resize width before inference")
    parser.add_argument("--resize-h", type=int, default=None, help="Resize height before inference")
    parser.add_argument("--no-window", action="store_true", help="Disable preview window")
    parser.add_argument("--fps-log-interval", type=float, default=2.0, help="Seconds between FPS logs")
    parser.add_argument("--gps-device", type=str, default=None, help="GPS serial device (e.g., /dev/ttyUSB0)")
    parser.add_argument("--gps-baud", type=int, default=9600, help="GPS serial baud rate")
    parser.add_argument("--gps-test", action="store_true", help="Only read GPS and print lat/lon")
    args = parser.parse_args()

    if args.gps_test:
        if not args.gps_device:
            raise SystemExit("Set --gps-device, e.g. /dev/ttyUSB0")
        gps_test(args.gps_device, args.gps_baud)

    process_camera(
        "camera_results",
        stride=args.stride,
        save_video=args.save_video,
        camera_id=args.camera_id,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        show_window=not args.no_window,
        fps_log_interval=args.fps_log_interval
    )
