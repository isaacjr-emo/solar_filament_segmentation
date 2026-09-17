"""
Solar Filament Tracking
=======================
Associate filament instances across a time series of H-alpha images.

Uses the existing segmentation pipeline, then links instances frame-to-frame
via IoU and/or centroid distance (optionally after differential-rotation
correction).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
from skimage import measure
import cv2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FilamentInstance:
    frame_idx: int
    label: int          # label inside this frame's labeled mask
    area_px: int
    centroid_y: float
    centroid_x: float
    length_px: float
    width_px: float
    elongation: float
    orientation: float
    bbox: Tuple[int, int, int, int]  # min_r, min_c, max_r, max_c
    mask: Optional[np.ndarray] = None  # binary mask of this instance (optional)


@dataclass
class Track:
    track_id: int
    instances: List[FilamentInstance] = field(default_factory=list)
    active: bool = True
    frames_since_seen: int = 0

    @property
    def start_frame(self) -> int:
        return self.instances[0].frame_idx if self.instances else -1

    @property
    def end_frame(self) -> int:
        return self.instances[-1].frame_idx if self.instances else -1

    @property
    def lifetime_frames(self) -> int:
        return self.end_frame - self.start_frame + 1 if self.instances else 0

    @property
    def max_area(self) -> int:
        return max((i.area_px for i in self.instances), default=0)

    @property
    def mean_area(self) -> float:
        if not self.instances:
            return 0.0
        return float(np.mean([i.area_px for i in self.instances]))

    def area_series(self) -> List[Tuple[int, int]]:
        return [(i.frame_idx, i.area_px) for i in self.instances]

    def centroid_series(self) -> List[Tuple[int, float, float]]:
        return [(i.frame_idx, i.centroid_y, i.centroid_x) for i in self.instances]


# ---------------------------------------------------------------------------
# Differential rotation (simple approximation)
# ---------------------------------------------------------------------------
def differential_rotation_shift(
    lat_deg: float,
    dt_days: float,
    image_shape: Tuple[int, int],
    disk_radius_px: float,
) -> float:
    """
    Approximate east-west shift in pixels due to differential rotation.

    Uses the standard solar differential rotation law (Snodgrass):
        ω (deg/day) ≈ 14.71 - 2.39 sin²φ - 1.78 sin⁴φ
    where φ is latitude.
    """
    # Sidereal rotation rate (deg/day)
    sin_lat = np.sin(np.deg2rad(lat_deg))
    omega = 14.71 - 2.39 * sin_lat**2 - 1.78 * sin_lat**4
    # Relative to Carrington rate ~14.18 deg/day (approx)
    carrington = 14.18
    d_lon_deg = (omega - carrington) * dt_days
    # Convert degrees of longitude at this latitude to pixels
    # Approximate: full disk diameter ≈ 180° of longitude at equator projection
    px_per_deg = (2 * disk_radius_px) / 180.0
    # Cosine foreshortening at latitude
    shift_px = d_lon_deg * px_per_deg * np.cos(np.deg2rad(lat_deg))
    return float(shift_px)


def pixel_to_lat(cy: float, img_h: int, disk_cy: float, disk_radius: float) -> float:
    """Rough latitude from pixel y (assuming disk centre known)."""
    # Normalized distance from equator (disk centre)
    dy = (disk_cy - cy) / (disk_radius + 1e-9)
    dy = np.clip(dy, -1.0, 1.0)
    return float(np.rad2deg(np.arcsin(dy)))


# ---------------------------------------------------------------------------
# Association costs
# ---------------------------------------------------------------------------
def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def pair_cost(
    inst_a: FilamentInstance,
    inst_b: FilamentInstance,
    max_centroid_dist: float,
    use_iou: bool = True,
    predicted_shift_x: float = 0.0,
) -> float:
    """
    Lower cost = better match.
    Returns a large number if the pair is incompatible.
    """
    # Centroid distance after optional differential-rotation prediction
    dx = (inst_b.centroid_x - predicted_shift_x) - inst_a.centroid_x
    dy = inst_b.centroid_y - inst_a.centroid_y
    dist = np.hypot(dx, dy)

    if dist > max_centroid_dist:
        return 1e6

    cost = dist / max_centroid_dist  # 0..1

    if use_iou and inst_a.mask is not None and inst_b.mask is not None:
        # Shift mask_a by predicted rotation for fairer IoU
        # (simple integer shift for speed)
        shift = int(round(predicted_shift_x))
        if shift != 0:
            shifted = np.roll(inst_a.mask, shift, axis=1)
        else:
            shifted = inst_a.mask
        iou = mask_iou(shifted, inst_b.mask)
        cost = 0.4 * cost + 0.6 * (1.0 - iou)

    return cost


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------
class FilamentTracker:
    """
    Online multi-object tracker for solar filaments.

    Parameters
    ----------
    max_centroid_dist : float
        Maximum pixel distance for a valid association.
    min_iou : float
        (Reserved for future hard IoU gate; currently soft via cost.)
    max_frames_lost : int
        How many consecutive frames a track may be missing before it is closed.
    use_differential_rotation : bool
        Apply a simple differential-rotation prediction before matching.
    dt_days_per_frame : float
        Time step between consecutive frames in days.
    disk_center : tuple (cy, cx)
        Approximate disk centre in pixels.
    disk_radius : float
        Approximate disk radius in pixels.
    """

    def __init__(
        self,
        max_centroid_dist: float = 40.0,
        min_iou: float = 0.1,
        max_frames_lost: int = 3,
        use_differential_rotation: bool = True,
        dt_days_per_frame: float = 0.01,
        disk_center: Optional[Tuple[float, float]] = None,
        disk_radius: Optional[float] = None,
    ):
        self.max_centroid_dist = max_centroid_dist
        self.min_iou = min_iou
        self.max_frames_lost = max_frames_lost
        self.use_differential_rotation = use_differential_rotation
        self.dt_days_per_frame = dt_days_per_frame
        self.disk_center = disk_center
        self.disk_radius = disk_radius

        self.tracks: List[Track] = []
        self._next_id = 1
        self._frame_idx = -1

    def _new_track(self, inst: FilamentInstance) -> Track:
        t = Track(track_id=self._next_id, instances=[inst], active=True)
        self._next_id += 1
        self.tracks.append(t)
        return t

    def update(self, instances: List[FilamentInstance], frame_idx: int) -> None:
        self._frame_idx = frame_idx

        active = [t for t in self.tracks if t.active]
        if not active:
            for inst in instances:
                self._new_track(inst)
            return

        if not instances:
            for t in active:
                t.frames_since_seen += 1
                if t.frames_since_seen > self.max_frames_lost:
                    t.active = False
            return

        n_tracks = len(active)
        n_dets = len(instances)
        cost = np.full((n_tracks, n_dets), 1e6, dtype=float)

        for i, track in enumerate(active):
            last = track.instances[-1]
            # Predicted shift due to differential rotation
            shift_x = 0.0
            if (
                self.use_differential_rotation
                and self.disk_center is not None
                and self.disk_radius is not None
            ):
                lat = pixel_to_lat(
                    last.centroid_y,
                    0,  # unused
                    self.disk_center[0],
                    self.disk_radius,
                )
                # frames elapsed since last observation
                dt_frames = frame_idx - last.frame_idx
                shift_x = differential_rotation_shift(
                    lat,
                    dt_frames * self.dt_days_per_frame,
                    (0, 0),
                    self.disk_radius,
                )

            for j, det in enumerate(instances):
                cost[i, j] = pair_cost(
                    last,
                    det,
                    max_centroid_dist=self.max_centroid_dist,
                    use_iou=True,
                    predicted_shift_x=shift_x,
                )

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e5:  # valid match
                active[r].instances.append(instances[c])
                active[r].frames_since_seen = 0
                matched_tracks.add(r)
                matched_dets.add(c)

        # Unmatched tracks → age or close
        for i, track in enumerate(active):
            if i not in matched_tracks:
                track.frames_since_seen += 1
                if track.frames_since_seen > self.max_frames_lost:
                    track.active = False

        # Unmatched detections → new tracks
        for j, det in enumerate(instances):
            if j not in matched_dets:
                self._new_track(det)

    def finalize(self) -> List[Track]:
        """Close all remaining active tracks and return the full list."""
        for t in self.tracks:
            t.active = False
        return self.tracks


# ---------------------------------------------------------------------------
# Helpers to turn segmentation output into instances
# ---------------------------------------------------------------------------
def labeled_mask_to_instances(
    labeled: np.ndarray,
    frame_idx: int,
    fil_info: Optional[List[dict]] = None,
    store_masks: bool = True,
) -> List[FilamentInstance]:
    """
    Convert a labeled mask (from detect_filaments) into FilamentInstance list.
    If fil_info is provided it is preferred for geometry; otherwise regionprops.
    """
    instances = []
    if fil_info:
        for f in fil_info:
            label = int(f["label"])
            mask = (labeled == label) if store_masks else None
            instances.append(
                FilamentInstance(
                    frame_idx=frame_idx,
                    label=label,
                    area_px=int(f["area_px"]),
                    centroid_y=float(f["centroid_y"]),
                    centroid_x=float(f["centroid_x"]),
                    length_px=float(f["length_px"]),
                    width_px=float(f["width_px"]),
                    elongation=float(f["elongation"]),
                    orientation=float(f["orientation"]),
                    bbox=tuple(f["bbox"]),
                    mask=mask,
                )
            )
    else:
        props = measure.regionprops(labeled)
        for p in props:
            try:
                mj = p.axis_major_length
                mn = p.axis_minor_length
            except AttributeError:
                mj = p.major_axis_length
                mn = p.minor_axis_length
            mask = (labeled == p.label) if store_masks else None
            instances.append(
                FilamentInstance(
                    frame_idx=frame_idx,
                    label=int(p.label),
                    area_px=int(p.area),
                    centroid_y=float(p.centroid[0]),
                    centroid_x=float(p.centroid[1]),
                    length_px=float(mj),
                    width_px=float(mn),
                    elongation=float(mj / (mn + 1e-9)),
                    orientation=float(p.orientation),
                    bbox=tuple(p.bbox),
                    mask=mask,
                )
            )
    return instances


def tracks_to_summary(tracks: List[Track]) -> List[dict]:
    """Flat summary table for display / CSV export."""
    rows = []
    for t in tracks:
        if not t.instances:
            continue
        areas = [i.area_px for i in t.instances]
        rows.append(
            {
                "track_id": t.track_id,
                "start_frame": t.start_frame,
                "end_frame": t.end_frame,
                "lifetime_frames": t.lifetime_frames,
                "n_observations": len(t.instances),
                "max_area_px": int(max(areas)),
                "mean_area_px": round(float(np.mean(areas)), 1),
                "min_area_px": int(min(areas)),
                "start_centroid_y": round(t.instances[0].centroid_y, 1),
                "start_centroid_x": round(t.instances[0].centroid_x, 1),
                "end_centroid_y": round(t.instances[-1].centroid_y, 1),
                "end_centroid_x": round(t.instances[-1].centroid_x, 1),
            }
        )
    return rows


def color_for_track(track_id: int, max_id: int = 50) -> Tuple[float, float, float]:
    """Deterministic bright colour for a track ID (for overlays)."""
    import matplotlib.pyplot as plt
    cmap = plt.cm.hsv
    return cmap((track_id % max_id) / max_id)[:3]


def build_track_overlay(
    img: np.ndarray,
    tracks: List[Track],
    frame_idx: int,
    alpha: float = 0.55,
) -> np.ndarray:
    """Colour-code the filaments that are present in this frame by track ID."""
    rgb = np.stack([img] * 3, axis=-1).astype(float)
    overlay = rgb.copy()

    for t in tracks:
        for inst in t.instances:
            if inst.frame_idx != frame_idx or inst.mask is None:
                continue
            col = color_for_track(t.track_id)
            for c in range(3):
                overlay[:, :, c][inst.mask] = col[c] * 200 + 55

    result = (1 - alpha) * rgb + alpha * overlay
    return np.clip(result, 0, 255).astype(np.uint8)
