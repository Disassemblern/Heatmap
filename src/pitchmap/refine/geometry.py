"""Pitch-space geometry for tracklet merging.

The vendored spatial check compares bounding boxes in image space, which says
almost nothing here: the camera pans and zooms, so the range of box positions
covers most of the frame and the check passes almost always. Calibration lets us
ask the question that actually matters instead - could one player physically
have travelled from the end of one fragment to the start of the next?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchmap.calib.homography import apply
from pitchmap.project.ground import foot_points


def pitch_positions(
    tracks: pd.DataFrame,
    matrices: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    """Foot point of every track row in pitch metres, row-aligned with tracks.

    Rows whose frame has no homography come back as NaN and are treated as
    unknown rather than as a constraint violation.
    """
    points = foot_points(tracks)
    frames = tracks["frame"].to_numpy(dtype=int)
    metres = np.full((len(tracks), 2), np.nan)

    for frame_index in np.unique(frames):
        if frame_index >= len(matrices) or source[frame_index] == "missing":
            continue
        rows = frames == frame_index
        metres[rows] = apply(matrices[frame_index], points[rows])

    return metres


def overlapping_rows(tracks: pd.DataFrame) -> np.ndarray:
    """Whether each row's box overlaps another box in the same frame, row-aligned.

    Image space is the right place for this question, unlike for merging: a
    tracker swaps identities where boxes touch, wherever the camera points.
    """
    boxes = tracks[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    touching = np.zeros(len(tracks), dtype=bool)
    for index in tracks.groupby("frame").indices.values():
        b = boxes[index]
        overlap = (
            (b[:, None, 0] < b[None, :, 2])
            & (b[None, :, 0] < b[:, None, 2])
            & (b[:, None, 1] < b[None, :, 3])
            & (b[None, :, 1] < b[:, None, 3])
        )
        np.fill_diagonal(overlap, False)
        touching[index] = overlap.any(axis=1)
    return touching


def required_speed(
    times_a: list[int],
    rows_a: list[int],
    times_b: list[int],
    rows_b: list[int],
    positions: np.ndarray,
    fps: float,
    max_samples: int = 400,
) -> float:
    """Speed a single player would need to be both fragments, in m/s.

    Evaluated at the pair of observations closest in time, which is where the
    demand is highest: two fragments seen a fraction of a second apart at
    opposite ends of the pitch cannot be one player, however plausible their
    endpoints look. Endpoint-only testing misses this, because merged tracklets
    interleave in time without sharing exact frames.

    Returns 0.0 when positions are unavailable, so missing calibration never
    blocks a merge.
    """
    first = _samples(times_a, rows_a, positions, max_samples)
    second = _samples(times_b, rows_b, positions, max_samples)
    if first is None or second is None:
        return 0.0

    times_first, points_first = first
    times_second, points_second = second

    # For each observation in A, the nearest in time in B.
    insert = np.searchsorted(times_second, times_first)
    best_speed = 0.0
    for side in (insert - 1, insert):
        valid = (side >= 0) & (side < len(times_second))
        if not valid.any():
            continue
        index_a = np.where(valid)[0]
        index_b = side[valid]

        gap_s = np.abs(times_first[index_a] - times_second[index_b]) / fps
        gap_s = np.maximum(gap_s, 1.0 / fps)
        distance = np.linalg.norm(points_first[index_a] - points_second[index_b], axis=1)
        speed = distance / gap_s

        closest = int(np.argmin(gap_s))
        best_speed = max(best_speed, float(speed[closest]))

    return best_speed


def _samples(
    times: list[int],
    rows: list[int],
    positions: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Time-sorted observations with a valid pitch position, thinned if long."""
    time_array = np.asarray(times)
    point_array = positions[np.asarray(rows)]
    keep = np.isfinite(point_array).all(axis=1)
    if not keep.any():
        return None

    time_array, point_array = time_array[keep], point_array[keep]
    order = np.argsort(time_array, kind="stable")
    time_array, point_array = time_array[order], point_array[order]

    if len(time_array) > max_samples:
        picks = np.linspace(0, len(time_array) - 1, max_samples).astype(int)
        time_array, point_array = time_array[picks], point_array[picks]

    return time_array, point_array
