"""Keyframe selection policy."""

from __future__ import annotations

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
