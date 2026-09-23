"""Render occupancy grids onto a football pitch.

The pipeline's origin is the top-left corner with y increasing downwards, while
mplsoccer's custom pitch puts the origin at the bottom left. The conversion is
confined to this module.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from mplsoccer import Pitch

from pitchmap.config import PitchConfig

GRASS = "#3f8f3f"
# Zero density is the grass itself, so the ramp starts at the pitch colour and
# the map blends into it instead of sitting on a black rectangle.
COLOUR_MAP = LinearSegmentedColormap.from_list(
    "pitch_heat",
    [GRASS, "#6cb043", "#a8cc4a", "#f7e34f", "#f9a83a", "#ef4123", "#c1121f"],
)
# Occupancy is long-tailed: a player standing still for a few seconds outweighs
# everywhere they ran. Compressing the scale lets the run pattern show instead
# of one saturated dot.
GAMMA = 0.45
# Saturate the top of the range rather than stretching to a single peak cell.
VMAX_PERCENTILE = 99.0


def _pitch(pitch: PitchConfig) -> Pitch:
    return Pitch(
        pitch_type="custom",
        pitch_length=pitch.length_m,
        pitch_width=pitch.width_m,
        line_color="#ffffff",
        pitch_color=GRASS,
        linewidth=1,
    )


def _draw(axis, grid: np.ndarray, pitch: PitchConfig) -> None:
    occupied = grid[grid > 0]
    vmax = float(np.percentile(occupied, VMAX_PERCENTILE)) if occupied.size else 1.0
    axis.imshow(
        np.flipud(grid),
        extent=(0.0, pitch.length_m, 0.0, pitch.width_m),
        origin="lower",
        cmap=COLOUR_MAP,
        norm=PowerNorm(gamma=GAMMA, vmin=0.0, vmax=vmax or 1.0),
        interpolation="bilinear",
        zorder=0.5,
    )


def render_single(grid: np.ndarray, pitch: PitchConfig, title: str, path: Path) -> None:
    figure, axis = _pitch(pitch).draw(figsize=(9, 6))
    _draw(axis, grid, pitch)
    axis.set_title(title, color="#c7d5cc", fontsize=11)
    figure.set_facecolor("#22312b")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def render_grid(
    entries: list[tuple[str, np.ndarray]],
    pitch: PitchConfig,
    path: Path,
    columns: int = 3,
) -> None:
    """Render several heatmaps as one contact sheet."""
    if not entries:
        return

    rows = int(np.ceil(len(entries) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 5.0, rows * 3.4))
    pitch_drawer = _pitch(pitch)

    for axis, (title, grid) in zip(np.atleast_1d(axes).ravel(), entries, strict=False):
        pitch_drawer.draw(ax=axis)
        _draw(axis, grid, pitch)
        axis.set_title(title, color="#c7d5cc", fontsize=9)

    for axis in np.atleast_1d(axes).ravel()[len(entries):]:
        axis.axis("off")

    figure.set_facecolor("#22312b")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
