"""Keyframe selection policy."""

from __future__ import annotations

import numpy as np

from pitchmap.io.video import VideoMeta


def uniform_keyframes(meta: VideoMeta, every_s: float) -> tuple[int, ...]:
    """Keyframes at a fixed spacing, always including the first and last frame."""
    stride = max(1, int(round(every_s * meta.fps)))
    frames = list(range(0, meta.n_frames, stride))
    last = meta.n_frames - 1
    if frames[-1] != last:
        frames.append(last)
    return tuple(frames)


def parse_keyframes(spec: str) -> tuple[int, ...]:
    """Parse an explicit comma-separated frame list."""
    return tuple(sorted({int(part) for part in spec.split(",") if part.strip()}))


def worst_drift_frames(
    drift_m: np.ndarray,
    source: np.ndarray,
    threshold_m: float,
    limit: int = 5,
) -> tuple[int, ...]:
    """Frames whose drift exceeds the threshold, worst first.

    These are the frames worth annotating next: adding a keyframe there splits
    the interval whose propagation is least trustworthy.
    """
    candidates = np.where((drift_m > threshold_m) & (source == "registered"))[0]
    if candidates.size == 0:
        return ()
    ordered = candidates[np.argsort(-drift_m[candidates])]
    return tuple(int(frame) for frame in ordered[:limit])
