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
import json
import os
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
    BEV_OBSTACLE_RADIUS_PX,
    BEV_HARD_BLOCK_DIST_M,
    MORPH_ENHANCED,
)
from data_logger import DataLogger
from scooter_commander import ScooterCommander
from gps_navigator import GPSNavigator
from object_detector import TinyObjectDetector, estimate_obstacle_distance
from bev_calibration import run_calibration, load_bev_params
from bev_obstacle import project_foot_to_bev, detection_to_metric, ObstacleEMAGrid
from stabilization import FrameStabilizer, TemporalMaskSmoother
from masks import split_masks, suppress_grass_in_mask, clean_sidewalk_mask, clean_bev_mask_enhanced, select_main_component, anchor_ego_to_mask, ego_connected_mask
from heading import heading_to_command, compute_speed, apply_planner_speed_limit
from visualization import draw_heading_hud, draw_bev_hud
from bev_predictor import BEVPredictiveTracker


KEY_LEFT = {81, 2424832}
KEY_UP = {82, 2490368}
KEY_RIGHT = {83, 2555904}
KEY_DOWN = {84, 2621440}


def _load_intent_schedule(schedule_path: str | None) -> list[dict]:
    """Load an optional frame-range intent schedule from JSON."""
    if not schedule_path:
        return []
    with open(schedule_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("events", [])
    if not isinstance(data, list):
        raise ValueError("Intent schedule must be a JSON list or an object with an 'events' list")

    schedule: list[dict] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        try:
            start_frame = int(item.get("start_frame"))
            end_frame = int(item.get("end_frame", start_frame))
        except Exception as exc:
            raise ValueError(f"Invalid frame bounds in schedule item {idx}: {item}") from exc
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        intent_raw = item.get("intent", None)
        intent = None if intent_raw is None else str(intent_raw).strip().lower()
        if intent in {"", "clear", "none", "null"}:
            intent = None
        if intent not in {None, "straight", "left", "right"}:
            raise ValueError(f"Invalid intent '{intent_raw}' in schedule item {idx}")
        schedule.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "intent": intent,
                "label": str(item.get("label", "")).strip(),
            }
        )
    schedule.sort(key=lambda item: (item["start_frame"], item["end_frame"]))
    return schedule


def _intent_event_for_frame(schedule: list[dict], frame_id: int, start_idx: int = 0) -> tuple[dict | None, int]:
    """Return the active schedule event for a frame, advancing a rolling index."""
    idx = max(0, int(start_idx))
    while idx < len(schedule) and frame_id > int(schedule[idx]["end_frame"]):
        idx += 1
    if idx < len(schedule):
        event = schedule[idx]
        if int(event["start_frame"]) <= frame_id <= int(event["end_frame"]):
            return event, idx
    return None, idx


# =============================================================================
# Main live loop
# =============================================================================
def run_live(camera_id=0, video_path=None, save_video=False, stride=1,
             enable_detection=True, gps_device=None, gps_waypoints=None,
             serial_port=None, enable_logging=False, log_dir="logs",
             headless=False, stab_radius=STAB_SMOOTHING_RADIUS,
             mask_alpha=MASK_SMOOTH_ALPHA, bev_alpha=BEV_SMOOTH_ALPHA,
             low_power=False, seg_input_res=SEG_INPUT_RES, path_scale=1.0,
             detection_stride=1, enable_predict=PREDICT_ENABLED,
             planner_mode="dijkstra", template_planner_enabled=True,
             max_frames=None, initial_intent=None, bev_clean_mode="auto",
             model_dir=None, output_video_path=None, seg_conf_thresh=0.5,
             enable_stabilization=STABILIZATION_ENABLED,
             enable_seg_smoothing=True,
             enable_bev_smoothing=True, enable_ego_anchor=True,
             enable_ego_connected=True,
             intent_schedule_path=None):
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
    enable_stabilization = bool(enable_stabilization)
    enable_seg_smoothing = bool(enable_seg_smoothing)
    enable_bev_smoothing = bool(enable_bev_smoothing)
    enable_ego_anchor = bool(enable_ego_anchor)
    enable_ego_connected = bool(enable_ego_connected)

    if low_power:
        stride = max(stride, LOW_POWER_STRIDE)
        detection_stride = max(detection_stride, LOW_POWER_DETECTION_STRIDE)
        path_scale = min(path_scale, LOW_POWER_PATH_SCALE)
        seg_input_res = (
            min(seg_input_res[0], LOW_POWER_SEG_INPUT_RES[0]),
            min(seg_input_res[1], LOW_POWER_SEG_INPUT_RES[1]),
        )
        print("[Perf] Low-power mode enabled.")
    bev_clean_mode = str(bev_clean_mode or "auto").strip().lower()
    if bev_clean_mode not in {"auto", "enhanced", "legacy"}:
        bev_clean_mode = "auto"
    use_enhanced_bev = bool(MORPH_ENHANCED)
    if bev_clean_mode == "enhanced":
        use_enhanced_bev = True
    elif bev_clean_mode == "legacy":
        use_enhanced_bev = False
    elif low_power:
        use_enhanced_bev = False
    print(f"[Perf] stride={stride}, det_stride={detection_stride}, "
          f"seg={seg_input_res[0]}x{seg_input_res[1]}, path_scale={path_scale:.2f}")
    print(f"[Perf] bev_clean={'enhanced' if use_enhanced_bev else 'legacy'}")
    planner_mode = str(planner_mode).strip().lower()
    if planner_mode not in {"dijkstra", "dfs"}:
        planner_mode = "dijkstra"
    print(f"[Planner] mode={planner_mode}")
    print(f"[Planner] template_planner={'on' if template_planner_enabled else 'off'}")
    effective_model_dir = str(model_dir).strip() if model_dir else MODEL_DIR

    # Initialize data logger
    logger = None
    if enable_logging:
        logger = DataLogger(log_dir=log_dir)

    # Initialize segmentation model
    print("Loading SegFormer model...")
    cfg = Config(
        model_dir=effective_model_dir,
        conf_thresh=float(seg_conf_thresh),
        road_id=ROAD_ID,
        inference_resize=seg_input_res,
        enable_logging=False,
    )
    seg_model = FastRoadDetector(cfg)
    print("SegFormer ready.")
    print(f"[Seg] model_dir={effective_model_dir}")
    print(f"[Seg] conf_thresh={float(seg_conf_thresh):.3f}")

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
    if enable_stabilization:
        stabilizer = FrameStabilizer(smoothing_radius=stab_radius)
        print(f"Frame stabilization ENABLED (radius={stab_radius}).")
    else:
        print("Frame stabilization DISABLED.")

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
    if not enable_seg_smoothing:
        print("Segmentation temporal smoothing DISABLED.")
    if not enable_bev_smoothing:
        print("BEV temporal smoothing DISABLED.")
    if not enable_ego_anchor:
        print("BEV ego anchor DISABLED.")
    if not enable_ego_connected:
        print("BEV ego-connected mask DISABLED.")

    # Initialize real-time BEV navigation stack
    work_grid = max(120, int(round(NAV_WORK_GRID_BASE * path_scale)))
    path_cfg = PathExtractorConfig(
        bev_forward_m=NAV_BEV_FORWARD_M,
        bev_lateral_m=NAV_BEV_LATERAL_M,
        work_size=(work_grid, work_grid),
        planner_mode=planner_mode,
        template_planner_enabled=bool(template_planner_enabled),
    )
    if low_power:
        path_cfg.search_max_depth = 5
        path_cfg.search_top_k = 4
    path_extractor = BEVPathExtractor(path_cfg)
    pp_controller = AdaptivePurePursuitController(PurePursuitConfig(dt_s=1.0 / 8.0))
    obstacle_ema = ObstacleEMAGrid(bev_h=BEV_SIZE[1], bev_w=BEV_SIZE[0])
    print(f"BEV planner ENABLED (work grid {work_grid}x{work_grid}).")

    # Initialize BEV predictive frame reuse
    predictor = BEVPredictiveTracker()
    if enable_predict:
        print("BEV predictive frame reuse ENABLED.")
    else:
        print("BEV predictive frame reuse DISABLED (except turn lock hold).")

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

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_w > 0 and frame_h > 0:
        print(f"[Video] frame_size={frame_w}x{frame_h}")

    # Load BEV calibration after capture open so it can scale to the active frame size.
    H_mat, Hinv, src_pts = load_bev_params(frame_size=(frame_w, frame_h))

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
    cached_frame_state = None

    scheduled_intent_family: str | None = str(initial_intent or "").strip().lower() or None
    if scheduled_intent_family not in {"straight", "left", "right"}:
        scheduled_intent_family = None
    base_intent_family = scheduled_intent_family
    manual_intent_family: str | None = None
    intent_schedule = _load_intent_schedule(intent_schedule_path)
    intent_schedule_idx = 0
    last_schedule_intent = "__unset__"
    turn_lock_family: str | None = None
    turn_lock_frames = 0
    turn_lock_release_streak = 0
    TURN_LOCK_ENTRY_DEG = 12.0
    TURN_LOCK_RELEASE_DEG = 9.0
    TURN_LOCK_MIN_FRAMES = 6
    TURN_LOCK_MAX_FRAMES = 22
    TURN_LOCK_RELEASE_STREAK = 2
    print("Running... Press q/ESC to quit | up=straight | left=left | right=right | down=clear intent\n")
    if scheduled_intent_family is not None:
        print(f"[Intent] initial={scheduled_intent_family}")
    if intent_schedule:
        print(f"[IntentSchedule] loaded {len(intent_schedule)} event(s) from {intent_schedule_path}")
    print(f"{'Frame':>6} | {'Command':>12} | {'Steer':>8} | {'Speed':>7} | "
          f"{'Obst':>5} | {'ms':>5} | {'FPS':>5} | {'IoU':>5} | {'Mode':>7}")
    print("-" * 92)

    try:
        while True:
            if max_frames is not None and frame_id >= int(max_frames):
                break
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

            scheduled_event, intent_schedule_idx = _intent_event_for_frame(
                intent_schedule,
                frame_id,
                intent_schedule_idx,
            )
            if scheduled_event is None:
                scheduled_intent_family = base_intent_family
                schedule_intent = None
            else:
                scheduled_intent_family = scheduled_event["intent"]
                schedule_intent = scheduled_intent_family
            schedule_token = "clear" if schedule_intent is None else schedule_intent
            if intent_schedule and schedule_token != last_schedule_intent:
                print(f"[IntentSchedule] frame={frame_id} intent={schedule_token}")
                last_schedule_intent = schedule_token

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
            obstacle_zones: list = []  # populated in compute branch; empty on skip frames
            active_intent_family = manual_intent_family if manual_intent_family in {"straight", "left", "right"} else scheduled_intent_family
            planner_intent_family = turn_lock_family if turn_lock_family in {"left", "right"} else active_intent_family
            turn_lock_active = bool(
                turn_lock_family in {"left", "right"}
                and predictor is not None
                and predictor.last_bev_mask is not None
                and predictor.last_path_model is not None
            )
            predict_skip = (
                turn_lock_active
                or (
                    enable_predict
                    and predictor is not None
                    and predictor.should_skip(speed=speed_est, dt=dt_frame)
                )
            )
            # Fallback: stride-based skip when predictor is disabled
            run_net = (not predict_skip) and (frame_id % stride == 0 or last_mask is None)
            if predict_skip:
                run_net = False
            hold_cached_frame = bool(
                video_path
                and stride > 1
                and not enable_predict
                and not run_net
                and not predict_skip
                and cached_frame_state is not None
            )

            if hold_cached_frame:
                # Aggressive video playback mode: hold the previous pipeline result
                # on skipped frames instead of rerunning BEV/path extraction.
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

                detections = last_detections
                min_obstacle_dist = last_min_obstacle_dist
                t_seg = 0.0
                t_det = 0.0
                t_bev = 0.0
                t_skel = 0.0
                t_path = 0.0

                bev_sidewalk = cached_frame_state["bev_sidewalk"]
                obstacle_zones = cached_frame_state["obstacle_zones"]
                nav_out = cached_frame_state["nav_out"]
                skel = cached_frame_state["skel"]
                paths = cached_frame_state["paths"]
                best_idx = cached_frame_state["best_idx"]
                best_path_len = cached_frame_state["best_path_len"]
                has_model_path = cached_frame_state["has_model_path"]
                graph_nodes = cached_frame_state["graph_nodes"]
                graph_edges = cached_frame_state["graph_edges"]
            elif predict_skip:
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
                if enable_bev_smoothing:
                    bev_sidewalk = bev_smoother.smooth(predicted_mask_255, flow_dy=bev_shift)
                else:
                    bev_sidewalk = predicted_mask_255
                t_bev = (time.time() - t_bev_start) * 1000

                # IMPORTANT: run graph planner on predicted BEV too, so Dijkstra/DFS
                # remains active on skip frames (not predictor-path-only).
                nav_out = path_extractor.process(
                    bev_sidewalk,
                    obstacle_zones_m=obstacle_zones if obstacle_zones else None,
                    gps_intent_family=planner_intent_family,
                    manual_intent_override=manual_intent_family in {"straight", "left", "right"},
                )
                t_skel = nav_out.t_skeleton_ms
                t_path = nav_out.t_path_ms

                # Keep predictor state aligned with planner output when available.
                if nav_out.has_path and nav_out.path_model is not None and nav_out.best_path_m is not None:
                    predictor.last_path_points_m = nav_out.best_path_m.copy()
                    predictor.last_path_model = nav_out.path_model
                    predictor.last_curvature = (
                        nav_out.path_model.curvature_of_x(1.0)
                        if nav_out.path_model.x_max_m >= 1.0
                        else 0.0
                    )

                paths = [(p, l) for p, l in zip(nav_out.candidate_paths_px, nav_out.candidate_lengths_m)]
                best_idx = nav_out.best_idx
                best_path_len_m = (
                    nav_out.candidate_lengths_m[best_idx]
                    if 0 <= best_idx < len(nav_out.candidate_lengths_m)
                    else 0.0
                )
                best_path_len = best_path_len_m * (bev_sidewalk.shape[0] / max(1e-6, NAV_BEV_FORWARD_M))
                graph_nodes = nav_out.graph_nodes
                graph_edges = nav_out.graph_edges

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
                if enable_seg_smoothing:
                    sidewalk_mask, seg_iou = seg_smoother.smooth(sidewalk_mask, return_iou=True)
                else:
                    seg_iou = 1.0
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
                    # --- Obstacle BEV projection + EMA update ---
                    foot_bev_pts = [project_foot_to_bev(d, H_mat, bev_h=BEV_SIZE[1], bev_w=BEV_SIZE[0])
                                    for d in detections]
                    obstacle_ema.update(foot_bev_pts)
                else:
                    detections = last_detections
                    min_obstacle_dist = last_min_obstacle_dist
                    obstacle_ema.update([])  # decay EMA on skipped detection frames
                t_det = (time.time() - t_det_start) * 1000

                # --- 3) BEV projection + cleaning ---
                t_bev_start = time.time()
                bev_sidewalk = cv2.warpPerspective(sidewalk_mask, H_mat, BEV_SIZE)
                if TRIM_BOTTOM > 0:
                    bev_sidewalk = bev_sidewalk[:-TRIM_BOTTOM, :]
                m_per_px = 0.5 * (
                    NAV_BEV_FORWARD_M / max(1.0, float(bev_sidewalk.shape[0] - 1))
                    + NAV_BEV_LATERAL_M / max(1.0, float(bev_sidewalk.shape[1] - 1))
                )
                if use_enhanced_bev:
                    bev_sidewalk = clean_bev_mask_enhanced(
                        bev_sidewalk,
                        m_per_px,
                        enhanced=True,
                    )
                else:
                    bev_sidewalk = clean_sidewalk_mask(bev_sidewalk, DT_CORE_THRESH)
                if enable_ego_anchor:
                    bev_sidewalk = anchor_ego_to_mask(bev_sidewalk, bev_size=BEV_SIZE)
                if enable_ego_connected:
                    bev_sidewalk = ego_connected_mask(bev_sidewalk, bev_size=BEV_SIZE)

                # Predictor: blend predicted + real on compute frames
                if predictor is not None:
                    # We need nav_out for the predictor, so extract path first
                    pass  # predictor blend happens after path extraction

                bev_shift = speed_est * dt_frame * BEV_PIXELS_PER_METER_FORWARD
                if enable_bev_smoothing:
                    bev_sidewalk = bev_smoother.smooth(bev_sidewalk, flow_dy=bev_shift)
                t_bev = (time.time() - t_bev_start) * 1000

                # --- 4) Path extraction (medial axis + graph + spline) ---
                # Hard-block: paint stop-zone obstacles black in BEV mask; build metric zones
                obstacle_zones: list = []
                for _d in detections:
                    _bx, _by = project_foot_to_bev(_d, H_mat, bev_h=BEV_SIZE[1], bev_w=BEV_SIZE[0])
                    if _d.get("distance_m", 999.0) < BEV_HARD_BLOCK_DIST_M:
                        _r_px = int(BEV_HARD_BLOCK_DIST_M * 50)
                        cv2.circle(bev_sidewalk, (int(_bx), int(_by)), _r_px, 0, -1)
                    _fwd, _lat = detection_to_metric(_d, H_mat, bev_sidewalk.shape)
                    _r_m = BEV_OBSTACLE_RADIUS_PX.get(_d.get("class_name", "default"),
                                                       BEV_OBSTACLE_RADIUS_PX["default"]) / 50.0
                    obstacle_zones.append((_fwd, _lat, _r_m))
                nav_out = path_extractor.process(
                    bev_sidewalk,
                    obstacle_zones_m=obstacle_zones if obstacle_zones else None,
                    gps_intent_family=planner_intent_family,
                    manual_intent_override=manual_intent_family in {"straight", "left", "right"},
                )
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
                has_model_path = bool(nav_out.has_path)
                graph_nodes = nav_out.graph_nodes
                graph_edges = nav_out.graph_edges

            if not hold_cached_frame:
                cached_frame_state = {
                    "bev_sidewalk": bev_sidewalk.copy(),
                    "obstacle_zones": list(obstacle_zones) if obstacle_zones else [],
                    "nav_out": nav_out,
                    "skel": skel.copy() if isinstance(skel, np.ndarray) else skel,
                    "paths": list(paths),
                    "best_idx": best_idx,
                    "best_path_len": best_path_len,
                    "has_model_path": has_model_path,
                    "graph_nodes": graph_nodes,
                    "graph_edges": graph_edges,
                }

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
            manual_control_active = manual_intent_family in {"straight", "left", "right"}
            manual_lookahead_scale = 1.0
            if manual_control_active:
                manual_lookahead_scale = 0.58 if manual_intent_family in {"left", "right"} else 0.75
            ctrl_out = pp_controller.update(
                nav_out.path_model if nav_out.has_path else None,
                speed_for_control,
                lookahead_scale=manual_lookahead_scale,
                disable_discont_hold=manual_control_active,
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

            if turn_lock_family in {"left", "right"}:
                turn_lock_frames += 1
                if turn_lock_frames >= TURN_LOCK_MIN_FRAMES:
                    if abs(heading_smoothed) <= TURN_LOCK_RELEASE_DEG:
                        turn_lock_release_streak += 1
                    else:
                        turn_lock_release_streak = 0
                if (
                    turn_lock_release_streak >= TURN_LOCK_RELEASE_STREAK
                    or turn_lock_frames >= TURN_LOCK_MAX_FRAMES
                ):
                    print(f"[TurnLock] released={turn_lock_family} frames={turn_lock_frames}")
                    turn_lock_family = None
                    turn_lock_frames = 0
                    turn_lock_release_streak = 0
            elif active_intent_family in {"left", "right"} and nav_out.has_path:
                desired_sign = -1.0 if active_intent_family == "left" else 1.0
                if desired_sign * heading_smoothed >= TURN_LOCK_ENTRY_DEG:
                    turn_lock_family = active_intent_family
                    turn_lock_frames = 0
                    turn_lock_release_streak = 0
                    print(f"[TurnLock] engaged={turn_lock_family}")

            has_candidate_path = bool(len(paths) > 0)
            has_model_path = bool(nav_out.has_path)
            has_control_path = bool(ctrl_out.valid_path)
            path_valid_for_speed = bool(has_model_path or has_control_path)
            speed_raw = compute_speed(heading_smoothed, min_obstacle_dist, path_valid_for_speed)
            speed_raw = apply_planner_speed_limit(
                speed_raw,
                getattr(nav_out, "suggested_slowdown", 0.0),
                getattr(nav_out, "is_low_confidence", False),
                cautious_speed_mps=SPEED_TURN,
            )
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
                    gps_intent_family=scheduled_intent_family or "",
                    planner_intent_family=planner_intent_family or "",
                    turn_lock_family=turn_lock_family or "",
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
                    has_candidate_path=int(has_candidate_path),
                    has_model_path=int(has_model_path),
                    has_control_path=int(has_control_path),
                    has_path=int(path_valid_for_speed),
                    num_paths=len(paths),
                    best_path_length_px=round(best_path_len, 1),
                    num_graph_nodes=graph_nodes,
                    num_graph_edges=graph_edges,
                    planner_mode=planner_mode,
                    path_source=getattr(nav_out, "path_source", ""),
                    bev_mask_occ_ratio=round(float(getattr(nav_out, "mask_occ_ratio", 0.0)), 5),
                    approval_confidence=round(float(getattr(nav_out, "approval_confidence", 0.0)), 4),
                    approval_margin=round(float(getattr(nav_out, "approval_margin", 0.0)), 4),
                    planner_low_confidence=int(bool(getattr(nav_out, "is_low_confidence", False))),
                    planner_slowdown=round(float(getattr(nav_out, "suggested_slowdown", 0.0)), 4),
                    selected_template_id=getattr(nav_out, "selected_template_id", ""),
                    selected_template_family=getattr(nav_out, "selected_template_family", ""),
                    corridor_confidence=round(float(getattr(nav_out, "corridor_confidence", 0.0)), 4),
                    corridor_valid_ratio=round(float(getattr(nav_out, "corridor_valid_ratio", 0.0)), 4),
                    corridor_forward_span_m=round(float(getattr(nav_out, "corridor_forward_span_m", 0.0)), 4),
                    corridor_width_cv=round(float(getattr(nav_out, "corridor_width_cv", 0.0)), 4),
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
                                           detections=detections, gps_info=gps_info,
                                           active_intent=planner_intent_family)

                bev_vis = draw_bev_hud(None, paths, best_idx, skel, bev_sidewalk,
                                       command, heading_smoothed, speed_smoothed,
                                       path_source=getattr(nav_out, "path_source", None),
                                       mask_occ_ratio=getattr(nav_out, "mask_occ_ratio", None),
                                       approval_confidence=getattr(nav_out, "approval_confidence", None),
                                       selected_template_family=getattr(nav_out, "selected_template_family", None),
                                       suggested_slowdown=getattr(nav_out, "suggested_slowdown", None),
                                       obstacle_zones_m=obstacle_zones if obstacle_zones else None,
                                       active_intent=planner_intent_family)

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
                    video_out = output_video_path or "heading_demo_output.mp4"
                    out_dir = os.path.dirname(video_out)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                    vw = cv2.VideoWriter(video_out, fourcc, actual_fps,
                                         (combined.shape[1], combined.shape[0]))
                    print(f"Saving output to {video_out}")
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
                key = cv2.waitKeyEx(1)
                if key == ord('q') or key == 27:
                    break
                elif key in KEY_UP:
                    manual_intent_family = 'straight'
                    if turn_lock_family is not None:
                        turn_lock_family = None
                        turn_lock_frames = 0
                        turn_lock_release_streak = 0
                    print(f"[Intent] straight")
                elif key in KEY_LEFT:
                    manual_intent_family = 'left'
                    if turn_lock_family not in {None, 'left'}:
                        turn_lock_family = None
                        turn_lock_frames = 0
                        turn_lock_release_streak = 0
                    print(f"[Intent] left")
                elif key in KEY_RIGHT:
                    manual_intent_family = 'right'
                    if turn_lock_family not in {None, 'right'}:
                        turn_lock_family = None
                        turn_lock_frames = 0
                        turn_lock_release_streak = 0
                    print(f"[Intent] right")
                elif key in KEY_DOWN:
                    manual_intent_family = None
                    if turn_lock_family is not None:
                        turn_lock_family = None
                        turn_lock_frames = 0
                        turn_lock_release_streak = 0
                    print(f"[Intent] cleared")

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
                model_dir=effective_model_dir, seg_resolution=seg_input_res,
                seg_conf_thresh=float(seg_conf_thresh),
                low_power=low_power, path_scale=path_scale,
                nav_work_grid=work_grid,
                nav_bev_forward_m=NAV_BEV_FORWARD_M,
                nav_bev_lateral_m=NAV_BEV_LATERAL_M,
                planner_mode=planner_mode,
                template_planner_enabled=bool(template_planner_enabled),
                max_frames=int(max_frames) if max_frames is not None else None,
                yolo_model=YOLO_MODEL_NAME if enable_detection else "disabled",
                output_video_path=output_video_path if save_video else None,
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
    parser.add_argument("--calib-frame", type=int, default=0,
                        help="Frame number to start calibration on (default: 0)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (1=all)")
    parser.add_argument("--save", action="store_true",
                        help="Save output video")
    parser.add_argument("--output-video", type=str, default=None,
                        help="Optional explicit path for saved output video")
    parser.add_argument("--low-power", action="store_true",
                        help="Enable small-computer profile (lower load, higher FPS)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Optional segmentation checkpoint directory or Hugging Face model id")
    parser.add_argument("--seg-conf-thresh", type=float, default=0.5,
                        help="Segmentation probability threshold for the drivable class")
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
    parser.add_argument("--no-stabilization", action="store_true",
                        help="Disable frame stabilization")
    parser.add_argument("--no-seg-smoothing", action="store_true",
                        help="Disable segmentation temporal smoothing")
    parser.add_argument("--no-bev-smoothing", action="store_true",
                        help="Disable BEV temporal smoothing")
    parser.add_argument("--no-ego-anchor", action="store_true",
                        help="Disable thin ego-to-mask bridge in BEV")
    parser.add_argument("--no-ego-connected", action="store_true",
                        help="Disable keeping only the ego-connected BEV component")
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
    parser.add_argument("--planner-mode", type=str, default="dijkstra",
                        choices=["dijkstra", "dfs"],
                        help="Graph candidate search mode for path planner")
    parser.add_argument("--no-template-planner", action="store_true",
                        help="Disable Phase 11 template approval and use graph/fallback baseline only")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional maximum number of processed frames")
    parser.add_argument("--intent", type=str, default=None,
                        choices=["straight", "left", "right"],
                        help="Initial persistent manual intent")
    parser.add_argument("--intent-schedule", type=str, default=None,
                        help="Optional JSON file with frame-range intent events")
    parser.add_argument("--bev-clean", type=str, default="auto",
                        choices=["auto", "enhanced", "legacy"],
                        help="BEV cleaning mode: auto uses enhanced except in low-power")

    # Display
    parser.add_argument("--headless", action="store_true",
                        help="No GUI window (for background/server runs). Use with --save.")

    args = parser.parse_args()

    if args.calibrate:
        run_calibration(camera_id=args.camera, video_path=args.video, start_frame=args.calib_frame)
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
            planner_mode=args.planner_mode,
            template_planner_enabled=not args.no_template_planner,
            max_frames=args.max_frames,
            initial_intent=args.intent,
            bev_clean_mode=args.bev_clean,
            model_dir=args.model_dir,
            output_video_path=args.output_video,
            seg_conf_thresh=args.seg_conf_thresh,
            enable_stabilization=not args.no_stabilization,
            enable_seg_smoothing=not args.no_seg_smoothing,
            enable_bev_smoothing=not args.no_bev_smoothing,
            enable_ego_anchor=not args.no_ego_anchor,
            enable_ego_connected=not args.no_ego_connected,
            intent_schedule_path=args.intent_schedule,
        )


