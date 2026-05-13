#!/usr/bin/env python3
"""
intent_picker.py
================
Interactive frame-by-frame intent picker.

Controls:
  SPACE       — advance to next frame
  s           — set intent: straight  (then fits path)
  l           — set intent: left      (then fits path)
  r           — set intent: right     (then fits path)
  c           — clear intent (free best)
  q / ESC     — quit

Usage:
  python intent_picker.py --video test_video_june_03_3.mp4
"""

import argparse
import sys
import cv2
import numpy as np

from fast_road_detector import FastRoadDetector, Config
from bev_calibration import load_bev_params
from masks import suppress_grass_in_mask, split_masks
from stabilization import TemporalMaskSmoother
from realtime_nav_core import BEVPathExtractor, PathExtractorConfig
from template_path_planner import (
    CorridorConfig, TemplatePlannerConfig,
    corridor_from_mask, approve_template_bank,
)
from config import MODEL_DIR, NAV_BEV_FORWARD_M as BEV_FORWARD_M, NAV_BEV_LATERAL_M as BEV_LATERAL_M, BEV_SIZE, SEG_INPUT_RES as SEG_RES


def load_pipeline():
    seg = FastRoadDetector(Config(model_dir=MODEL_DIR, conf_thresh=0.5, inference_resize=SEG_RES))
    H_mat, Hinv, src_pts = load_bev_params()
    smoother = TemporalMaskSmoother(alpha=0.65)
    extractor = BEVPathExtractor(PathExtractorConfig(
        bev_forward_m=BEV_FORWARD_M,
        bev_lateral_m=BEV_LATERAL_M,
        template_planner_enabled=True,
        template_planner_cfg=TemplatePlannerConfig(
            bev_forward_m=BEV_FORWARD_M,
            bev_lateral_m=BEV_LATERAL_M,
            corridor_cfg=CorridorConfig(
                bev_forward_m=BEV_FORWARD_M,
                bev_lateral_m=BEV_LATERAL_M,
            ),
        ),
    ))
    return seg, H_mat, smoother, extractor


def process_frame(frame, seg, H, smoother, extractor, intent):
    small = cv2.resize(frame, SEG_RES)
    mask_raw, _ = seg.process_frame(small, return_overlay=False)
    if mask_raw.shape != small.shape[:2]:
        mask_raw = cv2.resize(mask_raw, SEG_RES, interpolation=cv2.INTER_NEAREST)
    if mask_raw.ndim == 2:
        sidewalk = (mask_raw > 0).astype(np.uint8) * 255
    else:
        sidewalk, _ = split_masks(mask_raw)
    sidewalk = suppress_grass_in_mask(small, sidewalk)
    sidewalk, _ = smoother.smooth(sidewalk, return_iou=True)

    # Resize mask back to original frame resolution before warping
    # H was calibrated on the full-res frame, not SEG_RES
    fh, fw = frame.shape[:2]
    if sidewalk.shape != (fh, fw):
        sidewalk = cv2.resize(sidewalk, (fw, fh), interpolation=cv2.INTER_NEAREST)

    # Warp to BEV
    bev = cv2.warpPerspective(sidewalk, H, BEV_SIZE)
    nav = extractor.process(bev,
                            gps_intent_family=intent)
    return bev, nav


def draw_result(frame, bev, nav, intent):
    h, w = frame.shape[:2]
    bev_h, bev_w = bev.shape[:2] if bev.ndim == 2 else bev.shape[:2]

    # ---- Camera panel ----
    cam = frame.copy()
    label = f"Intent: {intent.upper() if intent else 'FREE'}  |  s=straight  l=left  r=right  c=clear  SPACE=next"
    cv2.rectangle(cam, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.putText(cam, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)

    # ---- BEV panel ----
    bev_vis = np.full((bev_h, bev_w, 3), (40, 40, 40), dtype=np.uint8)
    bev_gray = bev if bev.ndim == 2 else cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    bev_vis[bev_gray > 0] = (60, 150, 60)

    # Draw selected path
    if nav.has_path and nav.best_path_m is not None and len(nav.best_path_m) >= 2:
        path_px = nav.candidate_paths_px[nav.best_idx] if nav.best_idx < len(nav.candidate_paths_px) else None
        if path_px is not None and len(path_px) >= 2:
            pts = np.int32(path_px).reshape(-1, 1, 2)
            cv2.polylines(bev_vis, [pts], False, (0, 255, 0), 5, cv2.LINE_AA)
            cv2.circle(bev_vis, tuple(np.int32(path_px[0])), 7, (0, 255, 255), -1)
            cv2.circle(bev_vis, tuple(np.int32(path_px[-1])), 7, (0, 0, 255), -1)

    # Ego dot
    cv2.circle(bev_vis, (bev_w // 2, bev_h - 12), 10, (0, 200, 255), -1)

    # Info overlay
    fam   = nav.selected_template_family or '-'
    conf  = f"{nav.approval_confidence:.2f}"
    src   = nav.path_source
    is_tmpl = src == "template"
    is_reuse = src == "fallback_hold" and fam not in ('', '-')
    color = (0, 255, 100) if is_tmpl else (0, 180, 255) if is_reuse else (60, 60, 255)
    status = "APPROVED" if is_tmpl else ("REUSED" if is_reuse else "FALLBACK")
    cv2.rectangle(bev_vis, (0, 0), (bev_w, 72), (0, 0, 0), -1)
    cv2.putText(bev_vis, f"{status} | {fam}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(bev_vis, f"conf:{conf}  src:{src}", (6, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
    cv2.putText(bev_vis, f"slow:{nav.suggested_slowdown:.2f}", (6, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

    # Resize BEV to match camera height
    scale = h / bev_h
    bev_disp = cv2.resize(bev_vis, (int(bev_w * scale), h))

    return np.hstack([cam, bev_disp])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}")
        sys.exit(1)

    print("Loading pipeline...")
    seg, H, smoother, extractor = load_pipeline()
    print("Ready.\n")
    print("SPACE=next frame | s=straight | l=left | r=right | c=clear | q/ESC=quit\n")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    intent  = None
    frame   = None
    nav     = None
    bev     = None
    dirty   = True   # need to reprocess

    while True:
        # --- advance frame on SPACE ---
        if dirty:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break
            fn = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            print(f"[Frame {fn}] processing with intent={intent} ...")
            bev, nav = process_frame(frame, seg, H, smoother, extractor, intent)
            dirty = False

        disp = draw_result(frame, bev, nav, intent)
        cv2.imshow("Intent Picker", disp)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            dirty = True
        elif key == ord('s'):
            intent = 'straight'
            print(f"[Intent] straight")
            dirty = True
        elif key == ord('l'):
            intent = 'left'
            print(f"[Intent] left")
            dirty = True
        elif key == ord('r'):
            intent = 'right'
            print(f"[Intent] right")
            dirty = True
        elif key == ord('c'):
            intent = None
            print(f"[Intent] cleared")
            dirty = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
