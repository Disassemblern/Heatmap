"""Per-detection jersey colour features.

The torso sub-region of each tracked box is sampled, grass pixels are masked
out, and the median Lab colour of what remains is the feature.

Grass is identified by comparing against the pitch immediately around the
player rather than against a fixed green range. A fixed range cannot work here:
measured on the test clip it erased 100 percent of a green goalkeeper's shirt
and 62 percent of the navy kit, which reads as hue 55 under floodlight and so
falls inside any range wide enough to cover the grass. The grass beside a player
is the only reliable reference for what grass looks like in that part of the
frame, under that lighting, on that mowing stripe.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from pitchmap.io.video import iter_frames

# Torso window inside a player box: central band horizontally, upper body
# vertically (below the head, above the shorts).
TORSO_X = (0.25, 0.75)
TORSO_Y = (0.15, 0.50)
MIN_TORSO_PIXELS = 20
# Lab distance within which a torso pixel counts as the surrounding grass.
# Generous enough to cover the mowing stripes, tight enough to keep a green
# goalkeeper shirt, which sits well clear of the grass it stands on.
GRASS_LAB_DISTANCE = 22.0
REFERENCE_MARGIN = 0.6


@dataclass(frozen=True)
class JerseyFeatures:
    """Per-sampled-row jersey colours, aligned by raw table row index."""

    row_index: np.ndarray
    lab: np.ndarray
    n_pixels: np.ndarray


def sample_rows(tracks: pd.DataFrame, per_tracklet: int) -> pd.DataFrame:
    """Up to per_tracklet rows per tracklet, spread evenly over its life; 0 means every row."""
    def spread(group: pd.DataFrame) -> pd.DataFrame:
        if per_tracklet <= 0 or len(group) <= per_tracklet:
            return group
        positions = np.linspace(0, len(group) - 1, per_tracklet).astype(int)
        return group.iloc[positions]

    ordered = tracks.sort_values(["track_id", "frame"])
    return pd.concat(
        [spread(group) for _, group in ordered.groupby("track_id")]
    ).sort_values("frame")


def extract_features(video_path: str, sampled: pd.DataFrame) -> JerseyFeatures:
    """One sequential video pass extracting the jersey colour of each sampled row."""
    by_frame = {frame: group for frame, group in sampled.groupby("frame")}
    last_frame = int(sampled["frame"].max())

    row_indices: list[int] = []
    labs: list[np.ndarray] = []
    counts: list[int] = []

    for index, frame in iter_frames(video_path, stop=last_frame + 1):
        group = by_frame.get(index)
        if group is None:
            continue
        for row in group.itertuples():
            lab, n_pixels = _torso_colour(frame, row.x1, row.y1, row.x2, row.y2)
            row_indices.append(int(row.Index))
            labs.append(lab)
            counts.append(n_pixels)

    return JerseyFeatures(
        row_index=np.array(row_indices, dtype=np.int32),
        lab=np.stack(labs).astype(np.float32),
        n_pixels=np.array(counts, dtype=np.int32),
    )


def _torso_colour(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> tuple[np.ndarray, int]:
    height, width = frame.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    left = int(np.clip(x1 + TORSO_X[0] * box_w, 0, width - 1))
    right = int(np.clip(x1 + TORSO_X[1] * box_w, left + 1, width))
    top = int(np.clip(y1 + TORSO_Y[0] * box_h, 0, height - 1))
    bottom = int(np.clip(y1 + TORSO_Y[1] * box_h, top + 1, height))

    patch = frame[top:bottom, left:right]
    if patch.size == 0:
        return np.zeros(3, dtype=np.float32), 0

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    grass = _grass_reference(frame, x1, y1, x2, y2)
    if grass is None:
        keep = np.ones(lab.shape[:2], dtype=bool)
    else:
        keep = np.linalg.norm(lab - grass, axis=2) > GRASS_LAB_DISTANCE

    if keep.sum() < MIN_TORSO_PIXELS:
        return np.zeros(3, dtype=np.float32), 0

    return np.median(lab[keep], axis=0), int(keep.sum())


def _grass_reference(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> np.ndarray | None:
    """Median Lab of the pitch beside the player.

    Sampled from bands either side of the box, which are grass far more often
    than the band below it, where a second player's feet frequently sit.
    """
    height, width = frame.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    pad = max(4, int(REFERENCE_MARGIN * box_w))

    top = int(np.clip(y1, 0, height - 1))
    bottom = int(np.clip(y2, top + 1, height))
    bands = []
    for a, b in (
        (int(np.clip(x1 - pad, 0, width)), int(np.clip(x1, 0, width))),
        (int(np.clip(x2, 0, width)), int(np.clip(x2 + pad, 0, width))),
    ):
        if b - a >= 2:
            bands.append(frame[top:bottom, a:b].reshape(-1, 3))

    if not bands:
        return None
    pixels = np.concatenate(bands)
    if len(pixels) < MIN_TORSO_PIXELS:
        return None

    lab = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.median(lab.reshape(-1, 3), axis=0)
