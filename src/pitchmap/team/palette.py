"""Display colours for team labels.

Boxes are drawn in the team's own average kit colour rather than an arbitrary
palette, so a mislabelled player is visible as a box whose colour disagrees with
the shirt inside it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from pitchmap.team.assign import TEAM_A, TEAM_B, TEAM_OTHER, TEAM_UNKNOWN
from pitchmap.team.jersey import JerseyFeatures

FALLBACK = {
    TEAM_A: (255, 160, 60),
    TEAM_B: (60, 60, 255),
    TEAM_OTHER: (0, 255, 255),
    TEAM_UNKNOWN: (170, 170, 170),
}


def _lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    pixel = np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    bgr = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR).reshape(3)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def team_colours(
    labels: pd.DataFrame,
    tracks: pd.DataFrame,
    features: JerseyFeatures,
) -> dict[str, tuple[int, int, int]]:
    """Average kit colour per team label, falling back to a fixed palette."""
    colours = dict(FALLBACK)
    if labels.empty or len(features.row_index) == 0:
        return colours

    row_team = (
        tracks["track_id"]
        .iloc[features.row_index]
        .map(labels.set_index("track_id")["team"])
        .to_numpy()
    )
    usable = features.n_pixels > 0

    for team in (TEAM_A, TEAM_B, TEAM_OTHER):
        member = usable & (row_team == team)
        if member.sum() >= 5:
            colours[team] = _lab_to_bgr(np.median(features.lab[member], axis=0))
    return colours


def colour_by_track(
    labels: pd.DataFrame,
    colours: dict[str, tuple[int, int, int]],
) -> dict[int, tuple[int, int, int]]:
    """Box colour for every track id, keyed for the overlay drawing helpers."""
    return {
        int(row.track_id): colours.get(str(row.team), FALLBACK[TEAM_UNKNOWN])
        for row in labels.itertuples()
    }
