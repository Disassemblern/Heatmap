"""Accumulate pitch positions into occupancy grids.

Positions are binned onto a metric grid and smoothed once with a Gaussian
filter, which is equivalent to splatting a Gaussian per sample but far cheaper.
Each grid is normalised by its own sample count, so tracklets with different
visibility remain comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from pitchmap.config import HeatConfig, PitchConfig


def grid_shape(pitch: PitchConfig, cell_m: float) -> tuple[int, int]:
    return (int(round(pitch.width_m / cell_m)), int(round(pitch.length_m / cell_m)))


def occupancy(
    positions: pd.DataFrame,
    pitch: PitchConfig,
    heat: HeatConfig,
) -> np.ndarray:
    """Smoothed occupancy density over the pitch for one set of positions."""
    rows, columns = grid_shape(pitch, heat.cell_m)
    valid = positions[positions["valid"]]
    if valid.empty:
        return np.zeros((rows, columns))

    histogram, _, _ = np.histogram2d(
        valid["y_m"].to_numpy(),
        valid["x_m"].to_numpy(),
        bins=(rows, columns),
        range=((0.0, pitch.width_m), (0.0, pitch.length_m)),
    )

    smoothed = gaussian_filter(histogram, sigma=heat.sigma_m / heat.cell_m, mode="constant")
    total = smoothed.sum()
    return smoothed / total if total > 0 else smoothed


def on_pitch_tracklets(
    positions: pd.DataFrame,
    pitch: PitchConfig,
    margin_m: float = 0.5,
) -> set[int]:
    """Track ids whose typical position is on the pitch.

    Per-sample bounds cannot separate a coach from a player: the touchline
    tolerance has to stay generous because players legitimately step outside it
    for throw-ins and corners. A tracklet's median position does separate them,
    because a coach never spends the middle of their track on the pitch.

    Officials are not excluded by this: they are on the pitch like players, and
    removing them needs appearance, not geometry.
    """
    valid = positions[positions["valid"]]
    if valid.empty:
        return set()

    medians = valid.groupby("track_id")[["x_m", "y_m"]].median()
    inside = (
        medians["x_m"].between(-margin_m, pitch.length_m + margin_m)
        & medians["y_m"].between(-margin_m, pitch.width_m + margin_m)
    )
    return set(medians.index[inside])


def tracklet_summary(positions: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Per-tracklet visibility statistics, longest first."""
    grouped = positions.groupby("track_id")
    summary = pd.DataFrame(
        {
            "samples": grouped.size(),
            "valid_samples": grouped["valid"].sum(),
            "first_frame": grouped["frame"].min(),
            "last_frame": grouped["frame"].max(),
        }
    ).reset_index()

    return summary.assign(
        seconds=summary["valid_samples"] / fps,
        first_s=summary["first_frame"] / fps,
        last_s=summary["last_frame"] / fps,
    ).sort_values("seconds", ascending=False, ignore_index=True)
