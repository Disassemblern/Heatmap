"""Team box colours for the overlay and preview stages."""

from __future__ import annotations

import numpy as np

from pitchmap.io import artifacts
from pitchmap.team.jersey import JerseyFeatures
from pitchmap.team.palette import colour_by_track, team_colours


def load_track_colours(
    paths: artifacts.RunPaths,
    tracks,
) -> dict[int, tuple[int, int, int]] | None:
    """Per-track box colours from the team stage, or None when it has not run."""
    if not paths.team.exists() or not paths.team_features.exists():
        return None

    labels = artifacts.read_table(paths.team)
    source = paths.tracks_raw if paths.tracks_raw.exists() else paths.tracks
    feature_source = artifacts.read_table(source)

    with np.load(paths.team_features) as data:
        features = JerseyFeatures(
            row_index=data["row_index"], lab=data["lab"], n_pixels=data["n_pixels"]
        )

    colours = team_colours(labels, feature_source, features)
    return colour_by_track(labels, colours)
