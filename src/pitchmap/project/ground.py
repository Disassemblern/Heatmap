"""Project tracked boxes onto the pitch ground plane.

The foot point is the bottom centre of the box, which approximates where the
player meets the ground. Implausible results are flagged rather than dropped, so
the reason a position is missing stays visible downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchmap.calib.homography import apply
from pitchmap.config import PitchConfig

OUT_OF_BOUNDS_MARGIN_M = 3.0
MAX_SPEED_MS = 12.0
VALID = ""
REASON_NO_HOMOGRAPHY = "no_homography"
REASON_OUT_OF_BOUNDS = "out_of_bounds"
REASON_TELEPORT = "teleport"


def foot_points(tracks: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [(tracks["x1"] + tracks["x2"]) / 2.0, tracks["y2"]], axis=1
    ).astype(np.float64)


def project_tracks(
    tracks: pd.DataFrame,
    matrices: np.ndarray,
    source: np.ndarray,
    pitch: PitchConfig,
    fps: float,
) -> pd.DataFrame:
    """Project every tracked box to pitch metres and flag implausible results."""
    points = foot_points(tracks)
    frames = tracks["frame"].to_numpy(dtype=int)

    metres = np.full((len(tracks), 2), np.nan)
    for frame_index in np.unique(frames):
        if frame_index >= len(matrices) or source[frame_index] == "missing":
            continue
        rows = frames == frame_index
        metres[rows] = apply(matrices[frame_index], points[rows])

    projected = pd.DataFrame(
        {
            "frame": frames,
            "track_id": tracks["track_id"].to_numpy(dtype=int),
            "x_m": metres[:, 0],
            "y_m": metres[:, 1],
        }
    ).sort_values(["track_id", "frame"], ignore_index=True)

    return _flag(projected, pitch, fps)


def _flag(projected: pd.DataFrame, pitch: PitchConfig, fps: float) -> pd.DataFrame:
    reason = np.where(
        projected[["x_m", "y_m"]].isna().any(axis=1), REASON_NO_HOMOGRAPHY, VALID
    ).astype(object)

    outside = (
        (projected["x_m"] < -OUT_OF_BOUNDS_MARGIN_M)
        | (projected["x_m"] > pitch.length_m + OUT_OF_BOUNDS_MARGIN_M)
        | (projected["y_m"] < -OUT_OF_BOUNDS_MARGIN_M)
        | (projected["y_m"] > pitch.width_m + OUT_OF_BOUNDS_MARGIN_M)
    ).to_numpy()
    reason[(reason == VALID) & outside] = REASON_OUT_OF_BOUNDS

    speed = _speeds(projected, reason == VALID, fps)
    reason[(reason == VALID) & (speed > MAX_SPEED_MS)] = REASON_TELEPORT

    return projected.assign(valid=reason == VALID, invalid_reason=reason)


def _speeds(projected: pd.DataFrame, usable: np.ndarray, fps: float) -> np.ndarray:
    """Speed implied by the step from each row's previous usable sample."""
    speed = np.zeros(len(projected))
    positions = projected[["x_m", "y_m"]].to_numpy()
    track_ids = projected["track_id"].to_numpy()
    frames = projected["frame"].to_numpy()

    previous_track: int | None = None
    previous_index: int | None = None
    for index in range(len(projected)):
        if track_ids[index] != previous_track:
            previous_track, previous_index = track_ids[index], None
        if not usable[index]:
            continue
        if previous_index is not None:
            gap_s = max(frames[index] - frames[previous_index], 1) / fps
            distance = float(np.linalg.norm(positions[index] - positions[previous_index]))
            speed[index] = distance / gap_s
        previous_index = index

    return speed


def smooth_positions(positions: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Median-filter each track's valid positions to damp foot-point jitter."""
    smoothed = positions.copy()
    valid = positions["valid"].to_numpy()
    for column in ("x_m", "y_m"):
        values = positions[column].where(valid)
        smoothed[column] = (
            values.groupby(positions["track_id"])
            .transform(lambda series: series.rolling(window, center=True, min_periods=1).median())
            .where(valid, positions[column])
        )
    return smoothed
