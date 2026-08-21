"""Homography fitting from clicked landmarks, with error diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pitchmap.calib.pitch import landmarks
from pitchmap.config import PitchConfig

MIN_POINTS = 4


@dataclass(frozen=True)
class Fit:
    matrix: np.ndarray
    n_points: int
    n_inliers: int
    errors_m: np.ndarray
    names: tuple[str, ...]

    @property
    def median_error_m(self) -> float:
        return float(np.median(self.errors_m)) if self.errors_m.size else float("nan")

    @property
    def max_error_m(self) -> float:
        return float(np.max(self.errors_m)) if self.errors_m.size else float("nan")


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to an (N, 2) point array."""
    if points.size == 0:
        return points.reshape(0, 2)
    transformed = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float64), matrix.astype(np.float64)
    )
    return transformed.reshape(-1, 2)



def fit_points(
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    names: tuple[str, ...],
    ransac_reproj_px: float = 3.0,
) -> Fit:
    """Fit a pixel-to-metre homography from matched point arrays."""
    if len(image_points) < MIN_POINTS:
        raise ValueError(f"Need at least {MIN_POINTS} points, got {len(image_points)}")

    matrix, inliers = cv2.findHomography(
        image_points, pitch_points, cv2.RANSAC, ransacReprojThreshold=ransac_reproj_px
    )
    if matrix is None:
        raise ValueError("Homography fit failed; points may be collinear or mismatched")

    errors = np.linalg.norm(apply(matrix, image_points) - pitch_points, axis=1)
    return Fit(
        matrix=matrix,
        n_points=len(image_points),
        n_inliers=int(inliers.sum()) if inliers is not None else len(image_points),
        errors_m=errors,
        names=names,
    )


def fit_keyframe(
    clicks: dict[str, tuple[float, float]],
    pitch: PitchConfig,
    ransac_reproj_px: float = 3.0,
) -> Fit:
    """Fit a pixel-to-metre homography from named landmark clicks."""
    catalogue = landmarks(pitch)
    names = tuple(name for name in clicks if name in catalogue)
    if len(names) < MIN_POINTS:
        raise ValueError(f"Need at least {MIN_POINTS} landmarks, got {len(names)}")

    image_points = np.array([clicks[name] for name in names], dtype=np.float64)
    pitch_points = np.array([catalogue[name] for name in names], dtype=np.float64)
    return fit_points(image_points, pitch_points, names, ransac_reproj_px)


def holdout_errors_m(
    clicks: dict[str, tuple[float, float]],
    pitch: PitchConfig,
    ransac_reproj_px: float = 3.0,
) -> dict[str, float]:
    """Leave-one-out error per landmark: refit without it, then measure it.

    Unlike the in-fit residuals this cannot be driven to zero by overfitting, so
    it is the honest accuracy number for a keyframe.
    """
    catalogue = landmarks(pitch)
    usable = {name: point for name, point in clicks.items() if name in catalogue}
    if len(usable) <= MIN_POINTS:
        return {}

    errors: dict[str, float] = {}
    for held_out in usable:
        subset = {name: point for name, point in usable.items() if name != held_out}
        try:
            fit = fit_keyframe(subset, pitch, ransac_reproj_px)
        except ValueError:
            continue
        predicted = apply(fit.matrix, np.array([usable[held_out]], dtype=np.float64))[0]
        errors[held_out] = float(np.linalg.norm(predicted - np.array(catalogue[held_out])))

    return errors


def is_plausible(matrix: np.ndarray, width: int, height: int, pitch: PitchConfig) -> bool:
    """Reject homographies that fold, mirror, or degenerate.

    The test runs in pitch space, mapping the pitch outline into the image.
    Testing the image corners instead would fail on ordinary broadcast framing,
    where the top of the frame lies above the pitch horizon and legitimately
    projects to points at infinity.
    """
    if matrix is None or not np.all(np.isfinite(matrix)):
        return False
    if abs(np.linalg.det(matrix)) < 1e-12:
        return False

    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return False

    corners = np.array(
        [
            (0.0, 0.0),
            (pitch.length_m, 0.0),
            (pitch.length_m, pitch.width_m),
            (0.0, pitch.width_m),
        ],
        dtype=np.float64,
    )
    projected = apply(inverse, corners)
    if not np.all(np.isfinite(projected)):
        return False

    edges = np.diff(np.vstack([projected, projected[:1]]), axis=0)
    crosses = np.cross(edges, np.roll(edges, -1, axis=0))
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        return False

    # The pitch must land at a sane scale relative to the frame: neither a
    # speck nor spread across hundreds of screens.
    span = projected.max(axis=0) - projected.min(axis=0)
    diagonal = float(np.hypot(*span))
    frame_diagonal = float(np.hypot(width, height))
    return 0.2 * frame_diagonal <= diagonal <= 100.0 * frame_diagonal
