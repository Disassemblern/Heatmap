"""Fragmentation metrics: how far the tracks are from one-tracklet-per-person.

These need no annotation, so every tracking change gets a measured before/after
via the numbers printed by the track and refine stages and stored in meta.json.
"""

from __future__ import annotations

import pandas as pd

EXPECTED_PEOPLE = 22
LONG_TRACK_S = 20.0


def fragmentation_summary(
    tracks: pd.DataFrame,
    fps: float,
    n_frames: int,
    expected_people: int = EXPECTED_PEOPLE,
    long_s: float = LONG_TRACK_S,
) -> dict[str, float]:
    """Summary statistics of tracklet fragmentation.

    coverage is the identity-time captured: total tracked seconds divided by
    the seconds of person-presence a perfect tracker would capture. It can
    exceed 1.0 when non-players are tracked too.
    """
    if tracks.empty:
        return {
            "n_tracklets": 0,
            "n_long": 0,
            "median_s": 0.0,
            "mean_s": 0.0,
            "coverage": 0.0,
        }

    lengths_s = tracks.groupby("track_id").size() / fps
    ideal_seconds = (n_frames / fps) * expected_people
    return {
        "n_tracklets": int(len(lengths_s)),
        "n_long": int((lengths_s >= long_s).sum()),
        "median_s": float(lengths_s.median()),
        "mean_s": float(lengths_s.mean()),
        "coverage": float(lengths_s.sum() / ideal_seconds) if ideal_seconds else 0.0,
    }


def format_summary(summary: dict[str, float], long_s: float = LONG_TRACK_S) -> str:
    return (
        f"{summary['n_tracklets']} tracklets, {summary['n_long']} lasting >= {long_s:.0f} s, "
        f"median {summary['median_s']:.1f} s, mean {summary['mean_s']:.1f} s, "
        f"identity coverage {summary['coverage']:.2f}"
    )
