"""Contact sheets of tracklet crops, for judging whether a tracklet is one person."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchmap.io.video import iter_frames

CROP_W = 52
CROP_H = 124
LABEL_H = 22
PER_ROW = 6


def sample_crops(
    tracks: pd.DataFrame,
    track_ids: list[int],
    per_tracklet: int,
) -> dict[int, list[int]]:
    """Row indices to crop per tracklet, spread evenly across its life."""
    wanted: dict[int, list[int]] = {}
    for track_id in track_ids:
        group = tracks[tracks["track_id"] == track_id].sort_values("frame")
        if group.empty:
            continue
        picks = np.linspace(0, len(group) - 1, min(per_tracklet, len(group))).astype(int)
        wanted[track_id] = [int(i) for i in group.index.to_numpy()[picks]]
    return wanted


def render(
    video_path: str,
    tracks: pd.DataFrame,
    wanted: dict[int, list[int]],
    labels: dict[int, str],
    out_dir: Path,
    per_sheet: int = 24,
) -> list[Path]:
    """Write contact sheets, one strip of crops per tracklet.

    Crops are gathered in a single sequential pass over the video, because
    seeking per crop is far slower than decoding straight through.
    """
    needed = {row: track_id for track_id, rows in wanted.items() for row in rows}
    by_frame: dict[int, list[int]] = {}
    for row in needed:
        by_frame.setdefault(int(tracks["frame"].iloc[row]), []).append(row)

    crops: dict[int, np.ndarray] = {}
    last_frame = max(by_frame) if by_frame else 0
    for index, frame in iter_frames(video_path, stop=last_frame + 1):
        for row in by_frame.get(index, ()):
            crops[row] = _crop(frame, tracks.iloc[row])

    out_dir.mkdir(parents=True, exist_ok=True)
    strips = [
        _strip(track_id, rows, crops, labels.get(track_id, ""))
        for track_id, rows in sorted(wanted.items())
    ]
    strips = [strip for strip in strips if strip is not None]

    written = []
    for number, start in enumerate(range(0, len(strips), per_sheet), start=1):
        sheet = _grid(strips[start : start + per_sheet])
        path = out_dir / f"sheet_{number:02d}.png"
        cv2.imwrite(str(path), sheet)
        written.append(path)
    return written


def _crop(frame: np.ndarray, row: pd.Series) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = int(np.clip(row.x1, 0, width - 2))
    y1 = int(np.clip(row.y1, 0, height - 2))
    x2 = int(np.clip(row.x2, x1 + 1, width))
    y2 = int(np.clip(row.y2, y1 + 1, height))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
    return cv2.resize(patch, (CROP_W, CROP_H))


def _strip(
    track_id: int,
    rows: list[int],
    crops: dict[int, np.ndarray],
    label: str,
) -> np.ndarray | None:
    cells = [crops[row] for row in rows if row in crops]
    if not cells:
        return None
    body = np.hstack(cells)
    banner = np.zeros((LABEL_H, body.shape[1], 3), dtype=np.uint8)
    text = f"{track_id}" + (f" {label}" if label else "")
    cv2.putText(banner, text, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([banner, body])


def _grid(strips: list[np.ndarray]) -> np.ndarray:
    width = max(strip.shape[1] for strip in strips)
    padded = [np.pad(s, ((0, 0), (0, width - s.shape[1]), (0, 0))) for s in strips]
    rows = [
        np.hstack(padded[start : start + PER_ROW])
        for start in range(0, len(padded), PER_ROW)
    ]
    full = max(row.shape[1] for row in rows)
    rows = [np.pad(r, ((0, 0), (0, full - r.shape[1]), (0, 0))) for r in rows]
    return np.vstack(rows)
