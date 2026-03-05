#!/usr/bin/env python3
"""
live_heading_demo.py
====================
Live camera -> SegFormer -> BEV -> Skeleton -> Path -> Heading + Speed command.
+ YOLOv8-nano object detection (pedestrians, bicycles, cars)
+ GPS waypoint navigation (NMEA over serial)
+ Precise steering degree + speed output for scooter serial control

Run on MacBook with iPhone Continuity Camera (or any webcam).

Usage:
    python live_heading_demo.py                            # default camera 0
    python live_heading_demo.py --camera 1                 # iPhone camera
    python live_heading_demo.py --calibrate                # BEV calibration mode
    python live_heading_demo.py --video test.mp4           # test with video
    python live_heading_demo.py --gps-device COM3          # with GPS
    python live_heading_demo.py --gps-waypoints route.csv  # follow GPS route
    python live_heading_demo.py --serial-port COM4         # send commands to scooter
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

# ---- Detector (sidewalk segmentation) ----
from fast_road_detector import FastRoadDetector, Config
from realtime_nav_core import (
    BEVPathExtractor,
    PathExtractorConfig,
    AdaptivePurePursuitController,
    PurePursuitConfig,
)

# ---- Project modules ----
from config import (
    ROAD_ID, SIDEWALK_ID, MODEL_DIR,
    SEG_INPUT_RES, LOW_POWER_SEG_INPUT_RES,
    BEV_SIZE, TRIM_BOTTOM,
    DT_CORE_THRESH,
    YOLO_MODEL_NAME, YOLO_CONF_THRESH,
    STABILIZATION_ENABLED, STAB_SMOOTHING_RADIUS,
    MASK_SMOOTH_ALPHA, MASK_SMOOTH_CONSISTENCY_THRESH, BEV_SMOOTH_ALPHA,
    SEG_IOU_FAIL, SEG_IOU_WARN, SEG_FAIL_HOLD_FRAMES, SPEED_SEG_UNSTABLE,
    LOW_POWER_STRIDE, LOW_POWER_DETECTION_STRIDE, LOW_POWER_PATH_SCALE,
    NAV_BEV_FORWARD_M, NAV_BEV_LATERAL_M, NAV_WORK_GRID_BASE,
    BEV_PIXELS_PER_METER_FORWARD,
    GPS_STEER_GAIN, GPS_STEER_BIAS_MAX_DEG,
    SPEED_TURN,
    PREDICT_ENABLED,
)
from data_logger import DataLogger
from scooter_commander import ScooterCommander
from gps_navigator import GPSNavigator
from object_detector import TinyObjectDetector, estimate_obstacle_distance
from bev_calibration import run_calibration, load_bev_params
from stabilization import FrameStabilizer, TemporalMaskSmoother
from masks import split_masks, suppress_grass_in_mask, clean_sidewalk_mask, select_main_component, anchor_ego_to_mask, ego_connected_mask
from heading import heading_to_command, compute_speed
from visualization import draw_heading_hud, draw_bev_hud
from bev_predictor import BEVPredictiveTracker


# =============================================================================
# Main live loop
# =============================================================================
def run_live(camera_id=0, video_path=None, save_video=False, stride=1,
             enable_detection=True, gps_device=None, gps_waypoints=None,
             serial_port=None, enable_logging=False, log_dir="logs",
             headless=False, stab_radius=STAB_SMOOTHING_RADIUS,
             mask_alpha=MASK_SMOOTH_ALPHA, bev_alpha=BEV_SMOOTH_ALPHA,
             low_power=False, seg_input_res=SEG_INPUT_RES, path_scale=1.0,
             detection_stride=1, enable_predict=PREDICT_ENABLED):
    print("\n=== LIVE HEADING + OBJECT DETECTION + GPS DEMO ===")
    stab_radius = max(1, int(stab_radius))
    stride = max(1, int(stride))
    detection_stride = max(1, int(detection_stride))
    mask_alpha = float(np.clip(mask_alpha, 0.05, 0.95))
    bev_alpha = float(np.clip(bev_alpha, 0.05, 0.95))
    path_scale = float(np.clip(path_scale, 0.50, 1.0))
    seg_input_res = (
        max(192, int(seg_input_res[0])),
        max(128, int(seg_input_res[1])),
    )

    if low_power:
        stride = max(stride, LOW_POWER_STRIDE)
        detection_stride = max(detection_stride, LOW_POWER_DETECTION_STRIDE)
        path_scale = min(path_scale, LOW_POWER_PATH_SCALE)
        seg_input_res = (
            min(seg_input_res[0], LOW_POWER_SEG_INPUT_RES[0]),
            min(seg_input_res[1], LOW_POWER_SEG_INPUT_RES[1]),
        )
        print("[Perf] Low-power mode enabled.")
    print(f"[Perf] stride={stride}, det_stride={detection_stride}, "
          f"seg={seg_input_res[0]}x{seg_input_res[1]}, path_scale={path_scale:.2f}")

    # Initialize data logger
    logger = None
    if enable_logging:
        logger = DataLogger(log_dir=log_dir)

    # Load BEV calibration
    H_mat, Hinv, src_pts = load_bev_params()

    # Initialize segmentation model
    print("Loading SegFormer model...")
    cfg = Config(
        model_dir=MODEL_DIR,
        conf_thresh=0.5,
        road_id=ROAD_ID,
        inference_resize=seg_input_res,
    )
    seg_model = FastRoadDetector(cfg)
    print("SegFormer ready.")

    # Initialize object detector (YOLOv8-nano)
    obj_detector = None
    if enable_detection:
        print("Loading YOLOv8-nano object detector...")
        obj_detector = TinyObjectDetector(
            model_name=YOLO_MODEL_NAME,
            conf=YOLO_CONF_THRESH,
        )

    # Initialize GPS navigator
    gps_nav = None
    if gps_device or gps_waypoints:
        gps_nav = GPSNavigator(
            serial_device=gps_device,
            waypoints_file=gps_waypoints,
        )

    # Initialize scooter serial commander
    scooter = ScooterCommander(port=serial_port)

    # Initialize frame stabilizer (camera shake compensation)
    stabilizer = None
    if STABILIZATION_ENABLED:
        stabilizer = FrameStabilizer(smoothing_radius=stab_radius)
        print(f"Frame stabilization ENABLED (radius={stab_radius}).")

    # Initialize temporal mask smoothers
    seg_smoother = TemporalMaskSmoother(
        alpha=mask_alpha,
        consistency_thresh=MASK_SMOOTH_CONSISTENCY_THRESH,
    )
    bev_smoother = TemporalMaskSmoother(
        alpha=bev_alpha,
        consistency_thresh=0.35,
    )
    print(f"Temporal mask smoothing ENABLED (seg={mask_alpha:.2f}, bev={bev_alpha:.2f}).")

    # Initialize real-time BEV navigation stack
    work_grid = max(120, int(round(NAV_WORK_GRID_BASE * path_scale)))
    path_cfg = PathExtractorConfig(
        bev_forward_m=NAV_BEV_FORWARD_M,
        bev_lateral_m=NAV_BEV_LATERAL_M,
        work_size=(work_grid, work_grid),
    )
    if low_power:
        path_cfg.search_max_depth = 5
        path_cfg.search_top_k = 4
    path_extractor = BEVPathExtractor(path_cfg)
    pp_controller = AdaptivePurePursuitController(PurePursuitConfig(dt_s=1.0 / 8.0))
    print(f"BEV planner ENABLED (work grid {work_grid}x{work_grid}).")

    # Initialize BEV predictive frame reuse
    predictor = BEVPredictiveTracker() if enable_predict else None
    if predictor:
        print("BEV predictive frame reuse ENABLED.")

    # Open camera or video
    if video_path:
        # On Windows, iPhone .MOV is often HEVC and may "open" but fail to decode frames.
        # Try a few backends and validate by actually reading one frame.
        candidate_backends = []
        if hasattr(cv2, "CAP_FFMPEG"):
            candidate_backends.append(("FFMPEG", cv2.CAP_FFMPEG))
        if hasattr(cv2, "CAP_MSMF"):
            candidate_backends.append(("MSMF", cv2.CAP_MSMF))
        candidate_backends.append(("DEFAULT", None))

        cap = None
        for name, backend in candidate_backends:
            try:
                cap_try = (
                    cv2.VideoCapture(video_path, backend)
                    if backend is not None
                    else cv2.VideoCapture(video_path)
                )
            except Exception as e:
                print(f"[Video] Backend {name} failed: {e}")
                continue
            if not cap_try.isOpened():
                cap_try.release()
                continue
            ok, _ = cap_try.read()
            if not ok:
                cap_try.release()
                continue
            # Rewind so processing starts from frame 0.
            cap_try.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap = cap_try
            break

        if cap is None:
            # Final attempt (keeps prior behavior) to preserve any edge cases.
            cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF) if hasattr(cv2, "CAP_MSMF") else cv2.VideoCapture(video_path)
        print(f"Playing video: {video_path}")
    else:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"Camera {camera_id} opened.")

    if not cap.isOpened():
        print("ERROR: Cannot open camera/video.")
        return

    # Control smoothing buffers
    heading_buffer = deque(maxlen=3)
    speed_buffer = deque(maxlen=8)
    last_mask = None
    seg_iou = 1.0
    seg_unstable_frames = 0
    stability_mode = "NORMAL"
    last_detections = []
    last_min_obstacle_dist = None
    frame_id = 0
    fps_counter = deque(maxlen=30)
    fps = 0.0
    vw = None

    print("Running... Press 'q' or ESC to quit.\n")
    print(f"{'Frame':>6} | {'Command':>12} | {'Steer':>8} | {'Speed':>7} | "
          f"{'Obst':>5} | {'ms':>5} | {'FPS':>5} | {'IoU':>5} | {'Mode':>7}")
    print("-" * 92)

    try:
        while True:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                if video_path:
                    if frame_id == 0:
                        print(
                            "ERROR: Video opened but no frames could be decoded.\n"
                            "This is usually a codec issue (common with iPhone .MOV/HEVC on Windows).\n"
                            "Fix options:\n"
                            "  - Convert to H.264 MP4 (recommended), e.g. with ffmpeg:\n"
                            "      ffmpeg -i input.MOV -c:v libx264 -pix_fmt yuv420p -c:a aac output.mp4\n"
                            "  - Or install Windows HEVC Video Extensions.\n"
                            "  - Or record/export as 'Most Compatible' (H.264) instead of HEVC."
                        )
                    else:
                        print("End of video reached.")
                    break
                print("Frame lost, retrying...")
                continue

            frame_h, frame_w = frame.shape[:2]
            stab_corr_px = 0.0
            stab_corr_deg = 0.0

            # --- 0) Frame stabilization (camera shake compensation) ---
            raw_flow_dy = 0.0
            if stabilizer is not None:
                frame = stabilizer.stabilize(frame)
                stab_corr_px = stabilizer.last_correction_px
                stab_corr_deg = stabilizer.last_correction_deg
                raw_flow_dy = stabilizer.last_raw_dy

            # --- Adaptive prediction vs. full pipeline decision ---
            dt_frame = 1.0 / fps if fps > 0.5 else 0.13
            speed_est = speed_buffer[-1] if speed_buffer else 0.0
            heading_est = heading_buffer[-1] if heading_buffer else 0.0
            predict_skip = (
                predictor is not None
                and predictor.should_skip(speed=speed_est, dt=dt_frame)
            )
            # Fallback: stride-based skip when predictor is disabled
            run_net = (not predict_skip) and (frame_id % stride == 0 or last_mask is None)
            if predict_skip:
                run_net = False

            if predict_skip:
                # --- SKIP FRAME: predict BEV mask + path, no seg/path extraction ---
                t_seg_start = time.time()
                # Reuse last seg mask for visualization (no new segmentation)
                seg = last_mask
                if seg is not None:
                    if seg.ndim == 2 and seg.dtype == np.uint8:
                        sidewalk_mask = (seg > 0).astype(np.uint8) * 255
                        road_mask = np.zeros_like(sidewalk_mask)
                    else:
                        sidewalk_mask, road_mask = split_masks(seg)
                else:
                    sidewalk_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
                    road_mask = np.zeros_like(sidewalk_mask)
                t_seg = (time.time() - t_seg_start) * 1000

                # Object detection still runs on skip frames for safety
                t_det_start = time.time()
                run_detection = bool(obj_detector and (frame_id % detection_stride == 0))
                if run_detection:
                    detections = obj_detector.detect(frame)
                    for det in detections:
                        det["distance_m"] = estimate_obstacle_distance(det, frame_h)
                    min_obstacle_dist = min((d["distance_m"] for d in detections), default=None)
                    last_detections = detections
                    last_min_obstacle_dist = min_obstacle_dist
                else:
                    detections = last_detections
                    min_obstacle_dist = last_min_obstacle_dist
                t_det = (time.time() - t_det_start) * 1000

                # Predict BEV mask + path via affine warp (~1ms vs ~180ms)
                t_bev_start = time.time()
                predicted_mask_255, predicted_path_m = predictor.on_skip_frame(
                    speed=speed_est, dt=dt_frame, steer_deg=heading_est
                )
                bev_shift = speed_est * dt_frame * BEV_PIXELS_PER_METER_FORWARD
                bev_sidewalk = bev_smoother.smooth(predicted_mask_255, flow_dy=bev_shift)
                t_bev = (time.time() - t_bev_start) * 1000

                # Use predicted path directly — skip path extraction entirely
                t_skel = 0.0
                t_path = 0.0
                skel = np.zeros(BEV_SIZE[::-1], dtype=np.uint8)
                has_path = predicted_path_m is not None and len(predicted_path_m) >= 2
                if has_path:
                    # Build a lightweight PathPlanResult-like nav_out
                    from realtime_nav_core import PathPlanResult
                    nav_out = PathPlanResult(
                        has_path=True,
                        path_model=predictor.last_path_model,
                        best_path_m=predicted_path_m,
                        best_path_px=path_extractor._pixel_from_metric(
                            predicted_path_m, bev_sidewalk.shape
                        ),
                        candidate_paths_m=[predicted_path_m],
                        candidate_paths_px=[path_extractor._pixel_from_metric(
                            predicted_path_m, bev_sidewalk.shape
                        )],
                        candidate_lengths_m=[float(np.sum(np.sqrt(
                            np.sum(np.diff(predicted_path_m, axis=0) ** 2, axis=1)
                        ))) if len(predicted_path_m) >= 2 else 0.0],
                        best_idx=0,
                        skeleton_px=skel,
                        graph_nodes=0,
                        graph_edges=0,
                        t_skeleton_ms=0.0,
                        t_path_ms=0.0,
                    )
                else:
                    from realtime_nav_core import PathPlanResult
                    nav_out = PathPlanResult(
                        has_path=False,
                        path_model=None,
                        best_path_m=np.zeros((0, 2), dtype=np.float32),
                        best_path_px=np.zeros((0, 2), dtype=np.int32),
                        candidate_paths_m=[],
                        candidate_paths_px=[],
                        candidate_lengths_m=[],
                        best_idx=-1,
                        skeleton_px=skel,
                        graph_nodes=0,
                        graph_edges=0,
                        t_skeleton_ms=0.0,
                        t_path_ms=0.0,
                    )

                paths = [(p, l) for p, l in zip(nav_out.candidate_paths_px, nav_out.candidate_lengths_m)]
                best_idx = nav_out.best_idx
                best_path_len_m = (
                    nav_out.candidate_lengths_m[best_idx]
                    if 0 <= best_idx < len(nav_out.candidate_lengths_m)
                    else 0.0
                )
                best_path_len = best_path_len_m * (bev_sidewalk.shape[0] / max(1e-6, NAV_BEV_FORWARD_M))
                graph_nodes = 0
                graph_edges = 0

            else:
                # --- COMPUTE FRAME: full pipeline ---

                # --- 1) Segmentation ---
                t_seg_start = time.time()
                if run_net or last_mask is None:
                    seg, _ = seg_model.process_frame(frame, return_overlay=False)
                    if seg.shape != frame.shape[:2]:
                        seg = cv2.resize(seg, (frame.shape[1], frame.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                    last_mask = seg
                seg = last_mask
                if seg.ndim == 2 and seg.dtype == np.uint8:
                    sidewalk_mask = (seg > 0).astype(np.uint8) * 255
                    road_mask = np.zeros_like(sidewalk_mask)
                else:
                    sidewalk_mask, road_mask = split_masks(seg)
                sidewalk_mask = suppress_grass_in_mask(frame, sidewalk_mask)
                sidewalk_mask, seg_iou = seg_smoother.smooth(sidewalk_mask, return_iou=True)
                if seg_iou < SEG_IOU_FAIL:
                    seg_unstable_frames += 1
                elif seg_iou < SEG_IOU_WARN:
                    pass
                else:
                    seg_unstable_frames = max(0, seg_unstable_frames - 1)
                t_seg = (time.time() - t_seg_start) * 1000

                # --- 2) Object detection ---
                t_det_start = time.time()
                run_detection = bool(obj_detector and (frame_id % detection_stride == 0))
                if run_detection:
                    detections = obj_detector.detect(frame)
                    for det in detections:
                        det["distance_m"] = estimate_obstacle_distance(det, frame_h)
                    min_obstacle_dist = min((d["distance_m"] for d in detections), default=None)
                    last_detections = detections
                    last_min_obstacle_dist = min_obstacle_dist
                else:
                    detections = last_detections
                    min_obstacle_dist = last_min_obstacle_dist
                t_det = (time.time() - t_det_start) * 1000

                # --- 3) BEV projection + cleaning ---
                t_bev_start = time.time()
                bev_sidewalk = cv2.warpPerspective(sidewalk_mask, H_mat, BEV_SIZE)
                if TRIM_BOTTOM > 0:
                    bev_sidewalk = bev_sidewalk[:-TRIM_BOTTOM, :]
                bev_sidewalk = anchor_ego_to_mask(bev_sidewalk, bev_size=BEV_SIZE)
                bev_sidewalk = clean_sidewalk_mask(bev_sidewalk, DT_CORE_THRESH)
                bev_sidewalk = ego_connected_mask(bev_sidewalk, bev_size=BEV_SIZE)

                # Predictor: blend predicted + real on compute frames
                if predictor is not None:
                    # We need nav_out for the predictor, so extract path first
                    pass  # predictor blend happens after path extraction

                bev_shift = speed_est * dt_frame * BEV_PIXELS_PER_METER_FORWARD
                bev_sidewalk = bev_smoother.smooth(bev_sidewalk, flow_dy=bev_shift)
                t_bev = (time.time() - t_bev_start) * 1000

                # --- 4) Path extraction (medial axis + graph + spline) ---
                nav_out = path_extractor.process(bev_sidewalk)
                t_skel = nav_out.t_skeleton_ms
                t_path = nav_out.t_path_ms

                # Update predictor with real results
                if predictor is not None:
                    predictor.on_compute_frame(
                        real_bev_mask=bev_sidewalk,
                        path_result=nav_out,
                        speed=speed_est,
                        dt=dt_frame,
                        steer_deg=heading_est,
                    )

                skel = nav_out.skeleton_px
                paths = [(p, l) for p, l in zip(nav_out.candidate_paths_px, nav_out.candidate_lengths_m)]
                best_idx = nav_out.best_idx
                best_path_len_m = (
                    nav_out.candidate_lengths_m[best_idx]
                    if 0 <= best_idx < len(nav_out.candidate_lengths_m)
                    else 0.0
                )
                best_path_len = best_path_len_m * (bev_sidewalk.shape[0] / max(1e-6, NAV_BEV_FORWARD_M))
                has_path = bool(nav_out.has_path)
                graph_nodes = nav_out.graph_nodes
                graph_edges = nav_out.graph_edges

            # --- 6) GPS correction ---
            t_gps_start = time.time()
            gps_info = None
            gps_lat, gps_lon, gps_fix = None, None, 0
            gps_wp_name, gps_wp_dist, gps_correction = None, None, 0.0

            if gps_nav:
                gps_correction, gps_wp_dist, gps_wp_name = gps_nav.get_gps_heading_correction()
                gps_lat, gps_lon = gps_nav.get_position()
                with gps_nav.lock:
                    gps_fix = gps_nav.fix_quality
                gps_info = []
                if gps_lat is not None:
                    gps_info.append(f"GPS: {gps_lat:.6f}, {gps_lon:.6f}")
                if gps_wp_name:
                    gps_info.append(f"WP: {gps_wp_name} ({gps_wp_dist:.0f}m)" if gps_wp_dist else f"WP: {gps_wp_name}")
                    gps_info.append(f"GPS correction: {gps_correction:+.1f} deg")
            t_gps = (time.time() - t_gps_start) * 1000

            # --- 7) Adaptive pure pursuit + speed ---
            t_cmd_start = time.time()
            speed_for_control = speed_buffer[-1] if speed_buffer else SPEED_TURN
            ctrl_out = pp_controller.update(
                nav_out.path_model if nav_out.has_path else None,
                speed_for_control,
            )

            heading_raw = ctrl_out.steer_deg
            if gps_nav and gps_wp_name and gps_wp_name != "ARRIVED" and gps_correction is not None:
                gps_bias = float(np.clip(
                    gps_correction * GPS_STEER_GAIN,
                    -GPS_STEER_BIAS_MAX_DEG,
                    GPS_STEER_BIAS_MAX_DEG
                ))
                heading_raw = 0.8 * heading_raw + 0.2 * gps_bias

            heading_buffer.append(heading_raw)
            heading_smoothed = float(np.median(heading_buffer))
            command, cmd_color = heading_to_command(heading_smoothed)

            path_valid_for_speed = bool(has_path or ctrl_out.valid_path)
            speed_raw = compute_speed(heading_smoothed, min_obstacle_dist, path_valid_for_speed)
            speed_buffer.append(speed_raw)
            speed_smoothed = float(np.mean(speed_buffer))

            if seg_unstable_frames >= SEG_FAIL_HOLD_FRAMES:
                speed_smoothed = min(speed_smoothed, SPEED_SEG_UNSTABLE)
                stability_mode = "LIMITED"
            elif seg_iou < SEG_IOU_WARN:
                stability_mode = "WARN"
            else:
                stability_mode = "NORMAL"

            serial_str = scooter.send_command(heading_smoothed, speed_smoothed)
            t_cmd = (time.time() - t_cmd_start) * 1000

            # --- Total pipeline time ---
            t_total = (time.time() - t0) * 1000
            fps_counter.append(time.time())
            if len(fps_counter) > 1:
                fps = (len(fps_counter) - 1) / (fps_counter[-1] - fps_counter[0])
            else:
                fps = 0.0

            # --- 8) Log frame data ---
            if logger:
                det_classes = ";".join(d["class_name"] for d in detections) if detections else ""
                det_dists = ";".join(f'{d["distance_m"]:.2f}' for d in detections) if detections else ""
                logger.log(
                    frame_id=frame_id,
                    # Timing
                    t_segmentation=round(t_seg, 2),
                    t_detection=round(t_det, 2),
                    t_bev=round(t_bev, 2),
                    t_skeleton=round(t_skel, 2),
                    t_pathfinding=round(t_path, 2),
                    t_gps_fusion=round(t_gps, 2),
                    t_command=round(t_cmd, 2),
                    t_total_pipeline=round(t_total, 2),
                    fps=round(fps, 2),
                    # Heading & control
                    heading_raw_deg=round(heading_raw, 3),
                    heading_smoothed_deg=round(heading_smoothed, 3),
                    command=command,
                    speed_raw_mps=round(speed_raw, 3),
                    speed_smoothed_mps=round(speed_smoothed, 3),
                    serial_cmd=serial_str,
                    pp_lookahead_m=round(ctrl_out.lookahead_m, 3),
                    pp_kappa_cmd_m_inv=round(ctrl_out.kappa_cmd_m_inv, 4),
                    pp_target_x_m=round(ctrl_out.target_x_m, 3),
                    pp_target_y_m=round(ctrl_out.target_y_m, 3),
                    pp_valid_path=int(ctrl_out.valid_path),
                    seg_iou=round(seg_iou, 4),
                    seg_unstable_frames=seg_unstable_frames,
                    stability_mode=stability_mode,
                    stab_corr_px=round(stab_corr_px, 3),
                    stab_corr_deg=round(stab_corr_deg, 3),
                    # Path info
                    has_path=int(path_valid_for_speed),
                    num_paths=len(paths),
                    best_path_length_px=round(best_path_len, 1),
                    num_graph_nodes=graph_nodes,
                    num_graph_edges=graph_edges,
                    # Detections
                    num_detections=len(detections),
                    min_obstacle_dist_m=round(min_obstacle_dist, 2) if min_obstacle_dist else "",
                    detection_classes=det_classes,
                    detection_distances=det_dists,
                    # GPS
                    gps_lat=gps_lat if gps_lat else "",
                    gps_lon=gps_lon if gps_lon else "",
                    gps_fix_quality=gps_fix,
                    gps_wp_name=gps_wp_name or "",
                    gps_wp_dist_m=round(gps_wp_dist, 1) if gps_wp_dist else "",
                    gps_correction_deg=round(gps_correction, 2) if gps_correction else "",
                    # Mask stats
                    sidewalk_mask_pixels=int(np.count_nonzero(sidewalk_mask)),
                    bev_mask_pixels=int(np.count_nonzero(bev_sidewalk)),
                    skeleton_pixels=int(np.count_nonzero(skel)),
                )

            # --- 9) Visualization ---
            render_visuals = (not headless) or save_video
            if render_visuals:
                cam_vis = frame.copy()
                seg_overlay = np.zeros_like(cam_vis)
                seg_overlay[sidewalk_mask > 0] = (0, 220, 0)
                if road_mask is not None:
                    seg_overlay[road_mask > 0] = (255, 140, 0)
                cam_vis = cv2.addWeighted(seg_overlay, 0.50, cam_vis, 0.85, 0)

                if paths and best_idx >= 0:
                    best_path = paths[best_idx][0]
                    pts = np.array(best_path, dtype=np.float32).reshape(-1, 1, 2)
                    cam_pts = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)
                    cam_pts_int = np.int32(cam_pts).reshape(-1, 1, 2)
                    # Draw path with glow effect (thick dark outline + bright inner)
                    cv2.polylines(cam_vis, [cam_pts_int], False, (0, 0, 0), 14, cv2.LINE_AA)
                    cv2.polylines(cam_vis, [cam_pts_int], False, cmd_color, 8, cv2.LINE_AA)

                cam_vis = draw_heading_hud(cam_vis, command, heading_smoothed, speed_smoothed,
                                           cmd_color, fps, t_total,
                                           detections=detections, gps_info=gps_info)

                bev_vis = draw_bev_hud(None, paths, best_idx, skel, bev_sidewalk,
                                       command, heading_smoothed, speed_smoothed)

                display_h = 540
                cam_display = cv2.resize(cam_vis,
                    (int(display_h * cam_vis.shape[1] / cam_vis.shape[0]), display_h))
                bev_display = cv2.resize(bev_vis,
                    (int(display_h * bev_vis.shape[1] / bev_vis.shape[0]), display_h))

                combined = np.hstack((cam_display, bev_display))

                if not headless:
                    cv2.imshow("Live Heading + Detection + GPS", combined)

                if save_video and vw is None:
                    # Use actual processing FPS (estimate ~5) so playback matches real speed
                    actual_fps = fps if fps > 1 else 5.0
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vw = cv2.VideoWriter("heading_demo_output.mp4", fourcc, actual_fps,
                                         (combined.shape[1], combined.shape[0]))
                if vw:
                    vw.write(combined)

            # Console output (every 10 frames)
            if frame_id % 10 == 0:
                obs_str = f"{min_obstacle_dist:.1f}m" if min_obstacle_dist else "none"
                print(f"{frame_id:6d} | {command:>12s} | {heading_smoothed:+7.1f}d | "
                      f"{speed_smoothed:5.2f}ms | {obs_str:>5s} | {t_total:5.0f} | {fps:5.1f} | "
                      f"{seg_iou:5.2f} | {stability_mode:>7s}",
                      flush=True)

            if not headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            frame_id += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Save logger metadata and close
        if logger:
            logger.save_metadata(
                camera_id=camera_id, video_path=video_path,
                stride=stride, detection_enabled=enable_detection,
                detection_stride=detection_stride,
                gps_device=gps_device, gps_waypoints=gps_waypoints,
                serial_port=serial_port, total_frames=frame_id,
                model_dir=MODEL_DIR, seg_resolution=seg_input_res,
                low_power=low_power, path_scale=path_scale,
                nav_work_grid=work_grid,
                nav_bev_forward_m=NAV_BEV_FORWARD_M,
                nav_bev_lateral_m=NAV_BEV_LATERAL_M,
                yolo_model=YOLO_MODEL_NAME if enable_detection else "disabled",
            )
            logger.close()
        # Cleanup
        scooter.stop()
        if gps_nav:
            gps_nav.stop()
        cap.release()
        if vw:
            vw.release()
        cv2.destroyAllWindows()
        print(f"\nDone. Processed {frame_id} frames.")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live heading + object detection + GPS navigation demo")

    # Camera / video
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID (0=default, 1=iPhone usually)")
    parser.add_argument("--video", type=str, default=None,
                        help="Video file path (instead of live camera)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run BEV calibration mode")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (1=all)")
    parser.add_argument("--save", action="store_true",
                        help="Save output video")
    parser.add_argument("--low-power", action="store_true",
                        help="Enable small-computer profile (lower load, higher FPS)")
    parser.add_argument("--seg-width", type=int, default=SEG_INPUT_RES[0],
                        help="Segmentation input width")
    parser.add_argument("--seg-height", type=int, default=SEG_INPUT_RES[1],
                        help="Segmentation input height")
    parser.add_argument("--path-scale", type=float, default=1.0,
                        help="Scale for skeleton/path stage (0.5-1.0, lower=faster)")
    parser.add_argument("--detection-stride", type=int, default=1,
                        help="Run object detection every N processed frames")
    parser.add_argument("--stab-radius", type=int, default=STAB_SMOOTHING_RADIUS,
                        help="Frame stabilizer smoothing radius (higher = smoother, slower response)")
    parser.add_argument("--mask-alpha", type=float, default=MASK_SMOOTH_ALPHA,
                        help="Segmentation temporal EMA alpha (0-1)")
    parser.add_argument("--bev-alpha", type=float, default=BEV_SMOOTH_ALPHA,
                        help="BEV temporal EMA alpha (0-1)")

    # Object detection
    parser.add_argument("--no-detection", action="store_true",
                        help="Disable YOLOv8-nano object detection")

    # GPS
    parser.add_argument("--gps-device", type=str, default=None,
                        help="GPS serial device (e.g., COM3 or /dev/ttyUSB0)")
    parser.add_argument("--gps-baud", type=int, default=9600,
                        help="GPS baud rate (default: 9600)")
    parser.add_argument("--gps-waypoints", type=str, default=None,
                        help="CSV file with GPS waypoints: lat,lon[,name]")

    # Scooter serial
    parser.add_argument("--serial-port", type=str, default=None,
                        help="Serial port for scooter commands (e.g., COM4)")
    parser.add_argument("--serial-baud", type=int, default=115200,
                        help="Scooter serial baud rate (default: 115200)")

    # Data logging
    parser.add_argument("--log", action="store_true",
                        help="Enable per-frame CSV data logging for experiments")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for log files (default: logs/)")

    # BEV prediction
    parser.add_argument("--no-predict", action="store_true",
                        help="Disable BEV predictive frame reuse (run full pipeline every frame)")

    # Display
    parser.add_argument("--headless", action="store_true",
                        help="No GUI window (for background/server runs). Use with --save.")

    args = parser.parse_args()

    if args.calibrate:
        run_calibration(camera_id=args.camera, video_path=args.video)
    else:
        run_live(
            camera_id=args.camera,
            video_path=args.video,
            save_video=args.save,
            stride=args.stride,
            enable_detection=not args.no_detection,
            gps_device=args.gps_device,
            gps_waypoints=args.gps_waypoints,
            serial_port=args.serial_port,
            enable_logging=args.log,
            log_dir=args.log_dir,
            headless=args.headless,
            stab_radius=args.stab_radius,
            mask_alpha=args.mask_alpha,
            bev_alpha=args.bev_alpha,
            low_power=args.low_power,
            seg_input_res=(args.seg_width, args.seg_height),
            path_scale=args.path_scale,
            detection_stride=args.detection_stride,
            enable_predict=not args.no_predict,
        )
