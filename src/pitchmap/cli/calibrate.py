"""Estimate a pitch homography for every frame.

The default path is automatic: a pose model locates the standard pitch
landmarks in each frame and a homography is fitted from them, independently per
frame so error cannot accumulate. Manually clicked keyframes, if present, are
used to check that automatic result rather than to replace it.
"""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from pitchmap.calib import keypoints as kp
from pitchmap.calib.homography import apply, fit_keyframe, fit_points, is_plausible
from pitchmap.calib.propagate import SOURCE_MISSING, sample_grid
from pitchmap.config import load_config
from pitchmap.io import artifacts
from pitchmap.io.video import iter_frames

SOURCE_AUTO = "auto"
SOURCE_SMOOTHED = "smoothed"
DEFAULT_MODEL = "models/yolo-football-pitch-detection.pt"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pitch keypoint model weights")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument(
        "--min-keypoint-conf", type=float, default=0.5, help="Keypoint confidence floor"
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Frames used to median-smooth the homography sequence (0 disables)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N frames")
    parser.add_argument(
        "--reuse-keypoints",
        action="store_true",
        help="Refit from cached keypoints instead of re-running the model",
    )


def _detect_keypoints(detector, meta, n_frames: int) -> dict[int, kp.KeypointDetection]:
    return {
        index: detector.detect(frame)
        for index, frame in tqdm(
            iter_frames(meta.path, stop=n_frames), total=n_frames, unit="frame", desc="keypoints"
        )
    }


def _save_keypoints(paths, detections: dict[int, kp.KeypointDetection]) -> None:
    """Cache raw keypoints so fitting and smoothing can be retuned cheaply."""
    import pandas as pd

    rows = [
        {"frame": index, "keypoint": keypoint, "x": point[0], "y": point[1]}
        for index, detection in detections.items()
        for keypoint, point in detection.points.items()
    ]
    artifacts.write_table(pd.DataFrame.from_records(rows), paths.keypoints)


def _load_keypoints(paths, n_frames: int) -> dict[int, kp.KeypointDetection]:
    table = artifacts.read_table(paths.keypoints)
    grouped = {
        int(frame): kp.KeypointDetection(
            {int(row.keypoint): (float(row.x), float(row.y)) for row in group.itertuples()}
        )
        for frame, group in table.groupby("frame")
    }
    return {index: grouped.get(index, kp.KeypointDetection({})) for index in range(n_frames)}


def _fit_frame(detection, pitch, ransac_px):
    if detection.count < kp.MIN_KEYPOINTS:
        return None
    image, world = kp.correspondences(detection, pitch)
    names = tuple(str(index) for index in sorted(detection.points))
    try:
        return fit_points(image, world, names, ransac_px)
    except ValueError:
        return None


def _smooth(matrices: np.ndarray, source: np.ndarray, grid: np.ndarray, window: int) -> np.ndarray:
    """Median-smooth the homography sequence through projected control points.

    Per-frame fits jitter by a few centimetres as keypoints appear and vanish.
    Smoothing happens in projected metric space because homography matrix
    entries have no meaningful average.
    """
    if window < 3:
        return matrices

    import cv2

    usable = source != SOURCE_MISSING
    projected = np.full((len(matrices), len(grid), 2), np.nan)
    for index in np.where(usable)[0]:
        projected[index] = apply(matrices[index], grid)

    smoothed = matrices.copy()
    half = window // 2
    for index in np.where(usable)[0]:
        low, high = max(0, index - half), min(len(matrices), index + half + 1)
        neighbourhood = projected[low:high]
        valid = neighbourhood[~np.isnan(neighbourhood[:, 0, 0])]
        if len(valid) < 3:
            continue
        target = np.median(valid, axis=0)
        matrix, _ = cv2.findHomography(grid, target, method=0)
        if matrix is not None and np.all(np.isfinite(matrix)):
            smoothed[index] = matrix
    return smoothed


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)
    n_frames = min(meta.n_frames, args.limit) if args.limit else meta.n_frames

    if args.reuse_keypoints and paths.keypoints.exists():
        detections = _load_keypoints(paths, n_frames)
        print(f"Refitting from cached keypoints ({paths.keypoints})")
    else:
        detector = kp.PitchKeypointDetector(args.model, args.device, args.min_keypoint_conf)
        print(f"Calibrating with {detector.describe()}")
        detections = _detect_keypoints(detector, meta, n_frames)
        _save_keypoints(paths, detections)

    matrices = np.tile(np.eye(3), (n_frames, 1, 1))
    source = np.full(n_frames, SOURCE_MISSING, dtype=object)
    residual_m = np.full(n_frames, np.nan)
    n_points = np.zeros(n_frames, dtype=int)

    for index in range(n_frames):
        fit = _fit_frame(detections[index], config.pitch, config.calib.ransac_reproj_px)
        if fit is None or not is_plausible(fit.matrix, meta.width, meta.height, config.pitch):
            continue
        matrices[index] = fit.matrix
        source[index] = SOURCE_AUTO
        residual_m[index] = fit.median_error_m
        n_points[index] = fit.n_points

    grid = sample_grid(meta.width, meta.height)
    if args.smooth_window >= 3:
        matrices = _smooth(matrices, source, grid, args.smooth_window)
        source = np.where(source == SOURCE_AUTO, SOURCE_SMOOTHED, source).astype(object)

    artifacts.write_homographies(paths, matrices, source.astype(str), residual_m, n_points)
    _report(paths, config, n_frames, source, residual_m, n_points, matrices, grid)


def _check_against_keyframes(paths, config, matrices) -> None:
    """Compare the automatic result with any manually clicked keyframes."""
    manual = artifacts.read_keyframes(paths)
    if not manual:
        return

    print("\nAgreement with manually clicked keyframes:")
    for frame_index, clicks in sorted(manual.items()):
        if frame_index >= len(matrices):
            continue
        try:
            reference = fit_keyframe(clicks, config.pitch, config.calib.ransac_reproj_px)
        except ValueError:
            continue
        pixels = np.array(list(clicks.values()), dtype=np.float64)
        difference = np.linalg.norm(
            apply(matrices[frame_index], pixels) - apply(reference.matrix, pixels), axis=1
        )
        print(f"  frame {frame_index:6d}: median difference {np.median(difference):.2f} m")


def _report(paths, config, n_frames, source, residual_m, n_points, matrices, grid) -> None:
    covered = int((source != SOURCE_MISSING).sum())
    finite = residual_m[np.isfinite(residual_m)]
    counted = n_points[n_points > 0]

    print(f"\nCalibrated {covered}/{n_frames} frames ({n_frames - covered} without a fit)")
    if finite.size:
        print(
            f"Keypoint fit residual: median {np.median(finite):.2f} m, "
            f"90th percentile {np.percentile(finite, 90):.2f} m, worst {finite.max():.2f} m"
        )
    if counted.size:
        print(f"Keypoints used per frame: median {int(np.median(counted))}, min {counted.min()}")

    gaps = np.where(source == SOURCE_MISSING)[0]
    if gaps.size:
        print(f"Frames without calibration: {gaps[:10].tolist()}{' ...' if gaps.size > 10 else ''}")

    _check_against_keyframes(paths, config, matrices)

    artifacts.record_stage(
        paths,
        "calibrate",
        {
            "covered_frames": covered,
            "median_residual_m": float(np.median(finite)) if finite.size else None,
            "automatic": True,
        },
    )
    print(f"Artifact: {paths.homographies}")
