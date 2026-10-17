"""Team box colours for the overlay and preview stages."""

from __future__ import annotations

import numpy as np

from pitchmap.io import artifacts
from pitchmap.team.palette import colour_by_track, team_colours


def load_track_colours(paths: artifacts.RunPaths) -> dict[int, tuple[int, int, int]] | None:
    """Per-track box colours from the team stage, or None when it has not run."""
    if not paths.team.exists() or not paths.team_features.exists():
        return None

    with np.load(paths.team_features) as data:
        centroids = data["centroids"]
    return colour_by_track(artifacts.read_table(paths.team), team_colours(centroids))
