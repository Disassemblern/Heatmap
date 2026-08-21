"""Carry keyframe homographies across frames as the camera pans, tilts and zooms.

Each frame is registered *directly* to the keyframes that bracket it, using
features on the pitch plane. Direct registration is used rather than chaining
frame-to-frame motion because chaining accumulates error without bound: measured
on this footage, chaining drifts about 50 px over six seconds, while direct
registration to a keyframe stays sub-pixel for about ten seconds before feature
overlap runs out.

Where a frame registers to both bracketing keyframes, the two estimates are
blended by temporal weight and their disagreement is recorded as a drift
estimate in metres. Frames that register to neither fall back to chained motion
from their neighbour and are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pitchmap.calib.homography import apply

GREEN_LOWER = np.array([30, 40, 40], dtype=np.uint8)
GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)
MIN_MATCHES = 12
# A homography fitted to few correspondences reprojects its own inliers well
# while being badly wrong: on this footage a 15-inlier fit disagreed with a
# composed estimate by over 400 px. Demand both a healthy inlier count and a
# healthy inlier share before trusting a registration.
MIN_INLIERS = 25
MIN_INLIER_RATIO = 0.5
MAX_SCALE_CHANGE = 3.0
RATIO_TEST = 0.75
SOURCE_KEYFRAME = "keyframe"
SOURCE_REGISTERED = "registered"
SOURCE_CHAINED = "chained"
SOURCE_MISSING = "missing"


@dataclass(frozen=True)
class Registration:
    """Planar transform taking reference-frame pixels to target-frame pixels."""

    matrix: np.ndarray
    n_inliers: int
    reproj_px: float
    ok: bool


@dataclass(frozen=True)
class Features:
    keypoints: tuple
    descriptors: np.ndarray | None

    @property
    def usable(self) -> bool:
        return self.descriptors is not None and len(self.keypoints) >= MIN_MATCHES


FAILED = Registration(np.eye(3), 0, float("nan"), False)


def pitch_mask(frame: np.ndarray, boxes: np.ndarray | None = None) -> np.ndarray:
    """Mask selecting green pitch pixels, excluding detection boxes.

    Features are only usable for camera registration if they sit on the ground
    plane, so players, crowd and stands are excluded.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

    if boxes is not None:
        for x1, y1, x2, y2 in boxes.astype(int):
            pad = max(4, int(0.25 * (y2 - y1)))
            cv2.rectangle(
                mask,
                (max(0, x1 - pad), max(0, y1 - pad)),
                (min(mask.shape[1], x2 + pad), min(mask.shape[0], y2 + pad)),
                0,
                thickness=-1,
            )

    return mask


def build_detector(max_features: int = 3000) -> cv2.SIFT:
    return cv2.SIFT_create(nfeatures=max_features)


def extract(
    detector: cv2.SIFT,
    frame: np.ndarray,
    boxes: np.ndarray | None = None,
) -> Features:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, pitch_mask(frame, boxes))
    return Features(tuple(keypoints), descriptors)


def register(reference: Features, target: Features) -> Registration:
    """Estimate the transform from reference pixels to target pixels."""
    if not reference.usable or not target.usable:
        return FAILED

    matcher = cv2.BFMatcher()
    pairs = matcher.knnMatch(reference.descriptors, target.descriptors, k=2)
    good = [
        first
        for first, second in (pair for pair in pairs if len(pair) == 2)
        if first.distance < RATIO_TEST * second.distance
    ]
    if len(good) < MIN_MATCHES:
        return FAILED

    source = np.float32([reference.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    destination = np.float32([target.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    matrix, inliers = cv2.findHomography(source, destination, cv2.RANSAC, 3.0)
    if matrix is None or inliers is None or not np.all(np.isfinite(matrix)):
        return FAILED

    keep = inliers.reshape(-1) == 1
    if keep.sum() < MIN_INLIERS or keep.mean() < MIN_INLIER_RATIO:
        return FAILED
    if not _is_plausible_motion(matrix):
        return FAILED

    projected = cv2.perspectiveTransform(source, matrix).reshape(-1, 2)
    error = float(
        np.median(np.linalg.norm(projected[keep] - destination.reshape(-1, 2)[keep], axis=1))
    )
    return Registration(matrix, int(keep.sum()), error, True)


def _is_plausible_motion(matrix: np.ndarray) -> bool:
    """Reject transforms implying impossible camera motion between two views."""
    determinant = np.linalg.det(matrix[:2, :2])
    if not np.isfinite(determinant) or determinant <= 0:
        return False

    scale = float(np.sqrt(determinant))
    return 1.0 / MAX_SCALE_CHANGE <= scale <= MAX_SCALE_CHANGE


def transfer(anchor: np.ndarray, registration: Registration) -> np.ndarray:
    """Move an anchor homography onto a frame registered against that anchor.

    The anchor maps its own pixels to metres, so a target-frame pixel is first
    mapped back into anchor pixels before the anchor is applied.
    """
    moved = anchor @ np.linalg.inv(registration.matrix)
    return moved / moved[2, 2] if moved[2, 2] != 0 else moved


def sample_grid(width: int, height: int, steps: int = 6) -> np.ndarray:
    """Pixel sample points used to compare and blend two homographies."""
    xs = np.linspace(0.05 * width, 0.95 * width, steps)
    ys = np.linspace(0.05 * height, 0.95 * height, steps)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def blend(
    first: np.ndarray,
    second: np.ndarray,
    weight: float,
    grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Blend two homographies and report their disagreement in metres.

    Blending happens in projected metric space; homography matrix entries have
    no meaningful linear interpolation.
    """
    first_m = apply(first, grid)
    second_m = apply(second, grid)
    drift = float(np.median(np.linalg.norm(first_m - second_m, axis=1)))

    blended = (1.0 - weight) * first_m + weight * second_m
    matrix, _ = cv2.findHomography(grid, blended, method=0)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return (first if weight < 0.5 else second), drift
    return matrix, drift
