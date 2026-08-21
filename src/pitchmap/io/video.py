"""Video metadata and frame iteration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def read_meta(path: Path | str) -> VideoMeta:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        return VideoMeta(
            path=str(path),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def iter_frames(
    path: Path | str,
    start: int = 0,
    stop: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame) pairs sequentially from start to stop (exclusive)."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        if start:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while stop is None or index < stop:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        capture.release()


def read_frame(path: Path | str, index: int) -> np.ndarray:
    """Read a single frame by index. Slow for sequential access; use iter_frames instead."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise IndexError(f"Cannot read frame {index} from {path}")
        return frame
    finally:
        capture.release()
