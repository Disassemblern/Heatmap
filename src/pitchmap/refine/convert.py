"""Convert between the tracks table and GTA-Link Tracklet objects.

The vendored Tracklet container cannot carry provenance, so every tracklet is
accompanied by the list of raw table row indices it was built from. Split and
merge operate on those lists in parallel, and the final output is emitted by
row lookup into the raw table - geometry passes through bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitchmap.vendor.gta_link import Tracklet


@dataclass(frozen=True)
class TrackletSet:
    """GTA-Link tracklets plus per-tracklet raw row indices, keyed identically."""

    tracklets: dict[int, Tracklet]
    rows: dict[int, list[int]]


def to_tracklets(tracks: pd.DataFrame, feats: np.ndarray) -> TrackletSet:
    """Build Tracklet objects (frames, scores, tlwh boxes, features) per track id."""
    frames = tracks["frame"].to_numpy()
    scores = tracks["conf"].to_numpy()
    x1 = tracks["x1"].to_numpy()
    y1 = tracks["y1"].to_numpy()
    width = tracks["x2"].to_numpy() - x1
    height = tracks["y2"].to_numpy() - y1

    tracklets: dict[int, Tracklet] = {}
    rows: dict[int, list[int]] = {}
    for track_id, group in tracks.groupby("track_id"):
        indices = group.index.to_numpy()
        tracklet = Tracklet(
            track_id=int(track_id),
            frames=[int(frames[i]) for i in indices],
            scores=[float(scores[i]) for i in indices],
            bboxes=[
                [float(x1[i]), float(y1[i]), float(width[i]), float(height[i])]
                for i in indices
            ],
        )
        for index in indices:
            tracklet.append_feat(feats[index])
        tracklets[int(track_id)] = tracklet
        rows[int(track_id)] = [int(i) for i in indices]

    return TrackletSet(tracklets=tracklets, rows=rows)


def to_table(final: TrackletSet, raw: pd.DataFrame) -> pd.DataFrame:
    """Emit a tracks table from refined tracklets by raw-row lookup.

    Track ids are renumbered 1..N by first appearance so downstream output is
    tidy regardless of how many intermediate ids splitting created.
    """
    ordered = sorted(
        final.rows.items(), key=lambda item: min(raw["frame"].iloc[item[1]])
    )

    pieces = []
    for new_id, (_, indices) in enumerate(ordered, start=1):
        piece = raw.iloc[sorted(indices, key=lambda i: raw["frame"].iloc[i])].copy()
        piece["track_id"] = new_id
        pieces.append(piece)

    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["frame", "track_id"], ignore_index=True)
    )
