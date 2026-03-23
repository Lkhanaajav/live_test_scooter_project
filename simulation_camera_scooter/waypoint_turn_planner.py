"""
waypoint_turn_planner.py
========================
Phase 11.1: GPS-intent corridor waypoint turn planner.

Implements commanded-turn path planning via corridor-supported waypoint targets.

Algorithm:
  1. Extract commanded-side support from the corridor in a forward decision band.
  2. Form support clusters and choose a target from the midpoint of the
     best-supported cluster.
  3. Build a smooth entry -> apex -> exit path that stays controller-compatible.
  4. Gate the target first (support threshold), then gate the path (containment).
  5. Return either an approved path or an explicit low-confidence/hold result.

Design rules:
  - GPS or route logic provides maneuver intent; vision fits geometry.
  - Skeleton branches are NOT used as the primary turn selector.
  - This module is additive -- it does not modify realtime_nav_core.py.
  - All thresholds come from config.py (single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from template_path_planner import Corridor, CorridorConfig, corridor_from_mask

import config as cfg_module


# ---------------------------------------------------------------------------
# Configuration (reads from config.py single source of truth)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaypointTurnPlannerConfig:
    """Configuration for the waypoint-turn planner.

    All distances are in meters. Thresholds are dimensionless [0, 1].
    Defaults are read from config.py module constants.
    """
    # Forward decision band where turn openings are searched
    decision_band_min_m: float = cfg_module.WAYPOINT_DECISION_BAND_MIN_M
    decision_band_max_m: float = cfg_module.WAYPOINT_DECISION_BAND_MAX_M

    # Support thresholds for target acquisition and sustain
    acquire_support_min: float = cfg_module.WAYPOINT_ACQUIRE_SUPPORT_MIN
    sustain_support_min: float = cfg_module.WAYPOINT_SUSTAIN_SUPPORT_MIN

    # Target update hysteresis
    target_update_alpha: float = cfg_module.WAYPOINT_TARGET_UPDATE_ALPHA

    # Path containment thresholds
    path_containment_min: float = cfg_module.WAYPOINT_PATH_CONTAINMENT_MIN
    near_containment_min: float = cfg_module.WAYPOINT_NEAR_CONTAINMENT_MIN
    near_field_m: float = cfg_module.WAYPOINT_NEAR_FIELD_M

    # Confidence floor below which hold is recommended
    low_confidence_threshold: float = cfg_module.WAYPOINT_LOW_CONFIDENCE_THRESHOLD
    slowdown_floor: float = cfg_module.WAYPOINT_SLOWDOWN_FLOOR
    hold_slowdown: float = cfg_module.WAYPOINT_HOLD_SLOWDOWN

    # Path geometry
    path_horizon_m: float = cfg_module.WAYPOINT_PATH_HORIZON_M
    path_ds_m: float = cfg_module.WAYPOINT_PATH_DS_M
    exit_rejoin_factor: float = cfg_module.WAYPOINT_EXIT_REJOIN_FACTOR

    # BEV geometry (must match corridor config)
    bev_forward_m: float = 10.0
    bev_lateral_m: float = 10.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaypointTurnTarget:
    """A candidate waypoint target on the commanded side.

    Attributes
    ----------
    apex_m : tuple[float, float]
        (forward_m, lateral_m) of the turn apex in BEV metric coords.
    exit_m : tuple[float, float]
        (forward_m, lateral_m) of the exit anchor after the apex.
    support_score : float
        Fraction of corridor rows in the decision band that support this target.
    side : str
        "left" or "right".
    """
    apex_m: Tuple[float, float]
    exit_m: Tuple[float, float]
    support_score: float
    side: str


@dataclass
class WaypointTurnResult:
    """Result of the waypoint-turn planner for one frame.

    Attributes
    ----------
    active : bool
        True if the commanded-turn module produced a path (intent is left/right
        and corridor support was sufficient).
    intent : str
        The intent that was passed in ("left", "right", "straight", or "").
    target : WaypointTurnTarget | None
        The selected waypoint target, or None if inactive / unsupported.
    path_m : np.ndarray
        (N, 2) array of (forward_m, lateral_m) path points, or (0, 2) if inactive.
    confidence : float
        Overall confidence in [0, 1].
    recommend_hold : bool
        True if the planner recommends holding position (unsupported turn).
    suggested_slowdown : float
        Slowdown factor in [0, 1] (0 = full speed, 1 = stop).
    """
    active: bool
    intent: str
    target: Optional[WaypointTurnTarget]
    path_m: np.ndarray
    confidence: float
    recommend_hold: bool
    suggested_slowdown: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v to [lo, hi]."""
    return float(max(lo, min(hi, v)))


def _extract_commanded_side_support(
    corridor: Corridor,
    side: str,
    cfg: WaypointTurnPlannerConfig,
    bev_mask: np.ndarray,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Extract support clusters on the commanded side within the decision band.

    Scans the raw BEV mask (not just the corridor abstraction) to detect
    asymmetric turn openings on the commanded side. This avoids the problem
    where ``corridor_from_mask`` merges adjacent runs and hides side-specific
    openings.

    Returns
    -------
    support_score : float
        Fraction of decision-band rows that have commanded-side extent.
    clusters : list of (forward_m, lateral_m) midpoints
        Midpoints of contiguous commanded-side support regions in the band.
    """
    band_min = cfg.decision_band_min_m
    band_max = cfg.decision_band_max_m

    H, W = bev_mask.shape
    if H < 2 or W < 2:
        return 0.0, []

    m_per_px_fwd = cfg.bev_forward_m / max(1.0, float(H - 1))
    m_per_px_lat = cfg.bev_lateral_m / max(1.0, float(W - 1))

    # Import ego x position
    from config import bev_ego_x_px
    ego_x = bev_ego_x_px(W)

    # Convert decision band to pixel rows
    # forward_m = (H-1 - row) / (H-1) * bev_forward_m
    # row = H-1 - forward_m / bev_forward_m * (H-1)
    row_max = int(round(H - 1 - band_min / cfg.bev_forward_m * (H - 1)))
    row_min = int(round(H - 1 - band_max / cfg.bev_forward_m * (H - 1)))
    row_min = max(0, row_min)
    row_max = min(H - 1, row_max)

    mask_bin = (bev_mask > 0)

    # Scan decision-band rows for commanded-side support.
    # A commanded-side row has "support" if there are drivable pixels that
    # extend further on the commanded side than on the opposite side (asymmetric).
    supported_rows: List[Tuple[float, float]] = []
    # Count total scanned rows in the band (step by 4 to match scan stride)
    band_count = max(1, len(range(row_min, row_max + 1, max(1, int(4)))))
    min_asymmetry_px = max(3, int(0.3 / m_per_px_lat))  # ~0.3m in pixels

    for row in range(row_min, row_max + 1, max(1, int(4))):
        xs = np.where(mask_bin[row])[0]
        if len(xs) < 3:
            continue

        # Count pixels on each side of ego
        left_pixels = xs[xs < ego_x]
        right_pixels = xs[xs > ego_x]

        left_count = len(left_pixels)
        right_count = len(right_pixels)

        row_fwd = float(H - 1 - row) * m_per_px_fwd

        if side == "left":
            # Need more pixels on the left than on the right (asymmetric)
            if left_count > right_count + min_asymmetry_px and left_count >= 5:
                # Target lateral: midpoint of the leftmost commanded-side extent
                left_bound_lat = float(left_pixels[0] - ego_x) * m_per_px_lat
                mid_lat = left_bound_lat * 0.5  # midpoint between ego and left bound
                supported_rows.append((row_fwd, mid_lat))
        else:  # right
            if right_count > left_count + min_asymmetry_px and right_count >= 5:
                right_bound_lat = float(right_pixels[-1] - ego_x) * m_per_px_lat
                mid_lat = right_bound_lat * 0.5
                supported_rows.append((row_fwd, mid_lat))

    support_score = float(len(supported_rows)) / max(1, band_count)

    # Cluster contiguous supported rows and compute midpoints
    if not supported_rows:
        return support_score, []

    clusters = _cluster_support_rows(supported_rows)
    return support_score, clusters


def _cluster_support_rows(
    rows: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Group contiguous support rows into clusters and return their midpoints.

    Rows are sorted by forward distance. Two rows are in the same cluster if
    their forward gap is <= 0.5m.
    """
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda r: r[0])
    clusters: List[List[Tuple[float, float]]] = [[sorted_rows[0]]]

    for row in sorted_rows[1:]:
        if row[0] - clusters[-1][-1][0] <= 0.5:
            clusters[-1].append(row)
        else:
            clusters.append([row])

    # Return cluster midpoints, sorted by cluster size (best first)
    midpoints: List[Tuple[float, float]] = []
    for cluster in sorted(clusters, key=len, reverse=True):
        fwd_mid = float(np.mean([r[0] for r in cluster]))
        lat_mid = float(np.mean([r[1] for r in cluster]))
        midpoints.append((fwd_mid, lat_mid))

    return midpoints


def _select_target(
    side: str,
    support_score: float,
    clusters: List[Tuple[float, float]],
    corridor: Corridor,
    cfg: WaypointTurnPlannerConfig,
) -> Optional[WaypointTurnTarget]:
    """Select the best waypoint target from support clusters.

    Returns None if no cluster passes the target gate.
    """
    if not clusters or support_score < cfg.acquire_support_min:
        return None

    # Use the best (largest) cluster midpoint as the apex
    apex_fwd, apex_lat = clusters[0]

    # Compute exit anchor: further forward, blended back toward corridor center
    exit_fwd = min(apex_fwd + 2.0, cfg.path_horizon_m)

    # Find corridor center at the exit forward distance
    valid = corridor.valid_mask
    fwd = corridor.rows_forward_m
    valid_fwd = fwd[valid]
    valid_center = corridor.center_lateral_m[valid]

    if len(valid_fwd) >= 2:
        exit_center_lat = float(np.interp(exit_fwd, valid_fwd, valid_center))
    else:
        exit_center_lat = 0.0

    # Exit lateral blends from apex lateral toward corridor center
    exit_lat = (1.0 - cfg.exit_rejoin_factor) * apex_lat + cfg.exit_rejoin_factor * exit_center_lat

    return WaypointTurnTarget(
        apex_m=(apex_fwd, apex_lat),
        exit_m=(exit_fwd, exit_lat),
        support_score=support_score,
        side=side,
    )


def _build_turn_path(
    target: WaypointTurnTarget,
    cfg: WaypointTurnPlannerConfig,
) -> np.ndarray:
    """Build a smooth entry -> apex -> exit path to the waypoint target.

    Uses cubic Hermite interpolation with zero-tangent boundary conditions
    at ego (start straight) and smooth transition through apex to exit.
    This avoids the overshoot that polynomial fits produce.

    Returns (N, 2) array of (forward_m, lateral_m) points.
    """
    apex_fwd, apex_lat = target.apex_m
    exit_fwd, exit_lat = target.exit_m
    ds = max(0.05, cfg.path_ds_m)

    # Generate evenly spaced forward samples
    n_pts = max(10, int(exit_fwd / ds) + 2)
    fwd_samples = np.linspace(0.0, exit_fwd, n_pts, dtype=np.float64)

    # Use piecewise cubic Hermite: ego(0,0) with zero tangent -> apex -> exit
    # This gives a natural-looking path that starts straight then curves.
    #
    # Segment 1: ego to apex using cubic with zero start tangent
    # y(x) = a*t^3 + b*t^2  where t = x/apex_fwd, t in [0,1]
    # Boundary: y(0)=0, y'(0)=0, y(1)=apex_lat
    # => b*1 + a*1 = apex_lat, b = apex_lat - a
    # For a smooth curve, set a = -apex_lat (gives S-like shape: y = apex_lat*(3t^2 - 2t^3))
    #
    # Segment 2: apex to exit using linear blend (short segment)

    lat_samples = np.zeros_like(fwd_samples)

    for i, x in enumerate(fwd_samples):
        if apex_fwd > 0.01 and x <= apex_fwd:
            # Hermite segment: starts at (0,0) with zero tangent, ends at apex
            t = x / apex_fwd
            # Smooth step: 3t^2 - 2t^3 (starts flat, ends flat)
            lat_samples[i] = apex_lat * (3.0 * t * t - 2.0 * t * t * t)
        elif exit_fwd > apex_fwd + 0.01:
            # Linear blend from apex to exit
            t = (x - apex_fwd) / (exit_fwd - apex_fwd)
            t = min(1.0, max(0.0, t))
            lat_samples[i] = apex_lat + (exit_lat - apex_lat) * t
        else:
            lat_samples[i] = apex_lat

    path_m = np.stack([fwd_samples, lat_samples], axis=1).astype(np.float32)
    return path_m


def _gate_path_containment(
    path_m: np.ndarray,
    corridor: Corridor,
    cfg: WaypointTurnPlannerConfig,
) -> Tuple[float, float]:
    """Check what fraction of path points lie inside the corridor.

    Returns (containment_ratio, near_containment_ratio).
    """
    if path_m.shape[0] == 0:
        return 0.0, 0.0

    valid = corridor.valid_mask
    fwd = corridor.rows_forward_m
    valid_fwd = fwd[valid]
    valid_left = corridor.left_lateral_m[valid]
    valid_right = corridor.right_lateral_m[valid]

    if len(valid_fwd) < 2:
        return 0.0, 0.0

    path_fwd = path_m[:, 0].astype(np.float64)
    path_lat = path_m[:, 1].astype(np.float64)

    # Interpolate corridor boundaries at path forward positions
    left_at_path = np.interp(path_fwd, valid_fwd, valid_left)
    right_at_path = np.interp(path_fwd, valid_fwd, valid_right)

    # Check containment: path lateral must be between left and right
    # Allow small margin for numerical tolerance
    margin = 0.05  # 5cm tolerance
    inside = (path_lat >= left_at_path - margin) & (path_lat <= right_at_path + margin)

    # Also check that the path forward is within corridor coverage
    in_range = (path_fwd >= float(valid_fwd[0]) - 0.1) & (path_fwd <= float(valid_fwd[-1]) + 0.1)
    inside = inside & in_range

    containment = float(np.mean(inside)) if len(inside) > 0 else 0.0

    # Near-field containment
    near_mask = path_fwd <= cfg.near_field_m + 0.1
    if np.any(near_mask):
        near_containment = float(np.mean(inside[near_mask]))
    else:
        near_containment = containment

    return containment, near_containment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_waypoint_turn(
    corridor: Corridor,
    intent: Optional[str],
    bev_mask: np.ndarray,
    *,
    cfg: Optional[WaypointTurnPlannerConfig] = None,
    prev_target: Optional[WaypointTurnTarget] = None,
    prev_result: Optional[WaypointTurnResult] = None,
) -> WaypointTurnResult:
    """Plan a waypoint-conditioned turn path inside the BEV corridor.

    Parameters
    ----------
    corridor : Corridor
        Row-wise corridor geometry from ``corridor_from_mask()``.
    intent : str | None
        GPS/route maneuver intent: "left", "right", "straight", or None/"".
    bev_mask : np.ndarray
        Binary BEV mask (H, W) uint8, 255 = drivable.
    cfg : WaypointTurnPlannerConfig | None
        Configuration. Uses defaults if None.
    prev_target : WaypointTurnTarget | None
        Previous frame's selected target for hysteresis (Wave 2+).
    prev_result : WaypointTurnResult | None
        Previous frame's full result for continuity (Wave 2+).

    Returns
    -------
    WaypointTurnResult
        Turn planner output. ``active=False`` when intent is not a commanded
        turn or when the turn is unsupported.
    """
    cfg = cfg or WaypointTurnPlannerConfig()
    normalized_intent = str(intent or "").strip().lower()

    # No-intent or straight-intent: module stays inactive
    if normalized_intent not in ("left", "right"):
        return WaypointTurnResult(
            active=False,
            intent=normalized_intent,
            target=None,
            path_m=np.zeros((0, 2), dtype=np.float32),
            confidence=0.0,
            recommend_hold=False,
            suggested_slowdown=0.0,
        )

    # --- Commanded turn (left or right) ---

    # Step 1: Extract commanded-side support in the decision band
    support_score, clusters = _extract_commanded_side_support(
        corridor, normalized_intent, cfg, bev_mask
    )

    # Step 2: Select target from support clusters (target gate)
    target = _select_target(
        normalized_intent, support_score, clusters, corridor, cfg
    )

    if target is None:
        # Target gate failed: not enough corridor support on the commanded side
        return WaypointTurnResult(
            active=False,
            intent=normalized_intent,
            target=None,
            path_m=np.zeros((0, 2), dtype=np.float32),
            confidence=_clamp(0.15 * support_score, 0.0, 0.30),
            recommend_hold=True,
            suggested_slowdown=cfg.hold_slowdown,
        )

    # Step 3: Build smooth entry -> apex -> exit path
    path_m = _build_turn_path(target, cfg)

    # Step 4: Gate the path (containment check)
    containment, near_containment = _gate_path_containment(path_m, corridor, cfg)

    path_approved = bool(
        containment >= cfg.path_containment_min
        and near_containment >= cfg.near_containment_min
    )

    # Step 5: Compute confidence and return result
    if path_approved:
        # Confidence combines support score, containment, and corridor confidence
        confidence = _clamp(
            0.35 * support_score
            + 0.30 * containment
            + 0.20 * near_containment
            + 0.15 * corridor.confidence,
            0.0,
            1.0,
        )
        slowdown = _clamp(1.0 - confidence, 0.0, 1.0)

        return WaypointTurnResult(
            active=True,
            intent=normalized_intent,
            target=target,
            path_m=path_m,
            confidence=confidence,
            recommend_hold=False,
            suggested_slowdown=slowdown,
        )
    else:
        # Path gate failed: target was ok but the fitted path clips the corridor
        confidence = _clamp(
            0.20 * support_score + 0.10 * containment,
            0.0,
            0.35,
        )
        return WaypointTurnResult(
            active=False,
            intent=normalized_intent,
            target=target,
            path_m=np.zeros((0, 2), dtype=np.float32),
            confidence=confidence,
            recommend_hold=True,
            suggested_slowdown=max(cfg.slowdown_floor, _clamp(1.0 - confidence, 0.0, 1.0)),
        )
