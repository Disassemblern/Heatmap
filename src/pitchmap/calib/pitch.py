"""Metric model of a football pitch.

Coordinates are in metres with the origin at the top-left corner of a canonical
top-down view: x runs 0..length along the touchline, y runs 0..width across.
Landmark names are stable identifiers stored in keyframes.json, so renaming one
invalidates existing annotations.
"""

from __future__ import annotations

import numpy as np

from pitchmap.config import PitchConfig

PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_HALF_WIDTH_M = 20.16
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = 9.16
PENALTY_SPOT_DISTANCE_M = 11.0
CENTRE_CIRCLE_RADIUS_M = 9.15

Point = tuple[float, float]


def landmarks(pitch: PitchConfig) -> dict[str, Point]:
    """Named pitch landmarks in metres, ordered roughly left to right."""
    length, width = pitch.length_m, pitch.width_m
    mid_y = width / 2.0
    mid_x = length / 2.0
    points: dict[str, Point] = {}

    for side, base_x, direction in (("left", 0.0, 1.0), ("right", length, -1.0)):
        points[f"{side}_corner_top"] = (base_x, 0.0)
        points[f"{side}_corner_bottom"] = (base_x, width)

        box_x = base_x + direction * PENALTY_AREA_DEPTH_M
        points[f"{side}_penalty_top_goalline"] = (base_x, mid_y - PENALTY_AREA_HALF_WIDTH_M)
        points[f"{side}_penalty_bottom_goalline"] = (base_x, mid_y + PENALTY_AREA_HALF_WIDTH_M)
        points[f"{side}_penalty_top_corner"] = (box_x, mid_y - PENALTY_AREA_HALF_WIDTH_M)
        points[f"{side}_penalty_bottom_corner"] = (box_x, mid_y + PENALTY_AREA_HALF_WIDTH_M)

        goal_x = base_x + direction * GOAL_AREA_DEPTH_M
        points[f"{side}_goalarea_top_goalline"] = (base_x, mid_y - GOAL_AREA_HALF_WIDTH_M)
        points[f"{side}_goalarea_bottom_goalline"] = (base_x, mid_y + GOAL_AREA_HALF_WIDTH_M)
        points[f"{side}_goalarea_top_corner"] = (goal_x, mid_y - GOAL_AREA_HALF_WIDTH_M)
        points[f"{side}_goalarea_bottom_corner"] = (goal_x, mid_y + GOAL_AREA_HALF_WIDTH_M)

        points[f"{side}_penalty_spot"] = (base_x + direction * PENALTY_SPOT_DISTANCE_M, mid_y)

        # Where the penalty arc meets the penalty-area line.
        spot_to_box = PENALTY_AREA_DEPTH_M - PENALTY_SPOT_DISTANCE_M
        arc_offset = float(np.sqrt(CENTRE_CIRCLE_RADIUS_M**2 - spot_to_box**2))
        points[f"{side}_arc_top"] = (box_x, mid_y - arc_offset)
        points[f"{side}_arc_bottom"] = (box_x, mid_y + arc_offset)

    points["halfway_top"] = (mid_x, 0.0)
    points["halfway_bottom"] = (mid_x, width)
    points["centre_spot"] = (mid_x, mid_y)
    points["centre_circle_top"] = (mid_x, mid_y - CENTRE_CIRCLE_RADIUS_M)
    points["centre_circle_bottom"] = (mid_x, mid_y + CENTRE_CIRCLE_RADIUS_M)
    points["centre_circle_left"] = (mid_x - CENTRE_CIRCLE_RADIUS_M, mid_y)
    points["centre_circle_right"] = (mid_x + CENTRE_CIRCLE_RADIUS_M, mid_y)

    return points


def _arc(centre: Point, radius: float, start_deg: float, end_deg: float, steps: int = 40) -> np.ndarray:
    angles = np.radians(np.linspace(start_deg, end_deg, steps))
    return np.stack(
        [centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)], axis=1
    )


def line_segments(pitch: PitchConfig) -> tuple[np.ndarray, ...]:
    """Pitch markings as polylines in metres, for reprojection overlays."""
    length, width = pitch.length_m, pitch.width_m
    mid_y = width / 2.0
    mid_x = length / 2.0

    segments: list[np.ndarray] = [
        np.array([(0.0, 0.0), (length, 0.0), (length, width), (0.0, width), (0.0, 0.0)]),
        np.array([(mid_x, 0.0), (mid_x, width)]),
        _arc((mid_x, mid_y), CENTRE_CIRCLE_RADIUS_M, 0.0, 360.0),
    ]

    for base_x, direction in ((0.0, 1.0), (length, -1.0)):
        box_x = base_x + direction * PENALTY_AREA_DEPTH_M
        segments.append(
            np.array(
                [
                    (base_x, mid_y - PENALTY_AREA_HALF_WIDTH_M),
                    (box_x, mid_y - PENALTY_AREA_HALF_WIDTH_M),
                    (box_x, mid_y + PENALTY_AREA_HALF_WIDTH_M),
                    (base_x, mid_y + PENALTY_AREA_HALF_WIDTH_M),
                ]
            )
        )
        goal_x = base_x + direction * GOAL_AREA_DEPTH_M
        segments.append(
            np.array(
                [
                    (base_x, mid_y - GOAL_AREA_HALF_WIDTH_M),
                    (goal_x, mid_y - GOAL_AREA_HALF_WIDTH_M),
                    (goal_x, mid_y + GOAL_AREA_HALF_WIDTH_M),
                    (base_x, mid_y + GOAL_AREA_HALF_WIDTH_M),
                ]
            )
        )

        spot = (base_x + direction * PENALTY_SPOT_DISTANCE_M, mid_y)
        spot_to_box = PENALTY_AREA_DEPTH_M - PENALTY_SPOT_DISTANCE_M
        half_sweep = float(np.degrees(np.arccos(spot_to_box / CENTRE_CIRCLE_RADIUS_M)))
        centre_angle = 0.0 if direction > 0 else 180.0
        segments.append(
            _arc(
                spot,
                CENTRE_CIRCLE_RADIUS_M,
                centre_angle - half_sweep,
                centre_angle + half_sweep,
            )
        )

    return tuple(segments)
