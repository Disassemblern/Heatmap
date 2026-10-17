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


def team_colours(centroids: np.ndarray) -> dict[str, tuple[int, int, int]]:
    """Each team's fitted kit colour; `other` keeps a fixed colour unlike either kit.

    Taken from the centroids rather than re-averaged from rows: the colour rows
    are keyed to tracks_raw, the labels to the refined ids, and joining them by
    position paints both teams the same muddy mix.
    """
    return {
        **FALLBACK,
        TEAM_A: _lab_to_bgr(centroids[0]),
        TEAM_B: _lab_to_bgr(centroids[1]),
    }


def colour_by_track(
    labels: pd.DataFrame,
    colours: dict[str, tuple[int, int, int]],
) -> dict[int, tuple[int, int, int]]:
    """Box colour for every track id, keyed for the overlay drawing helpers."""
    return {
        int(row.track_id): colours.get(str(row.team), FALLBACK[TEAM_UNKNOWN])
        for row in labels.itertuples()
    }
