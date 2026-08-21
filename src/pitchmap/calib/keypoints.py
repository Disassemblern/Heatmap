"""Automatic per-frame pitch calibration from detected pitch keypoints.

A pose model locates the 32 standard pitch landmarks in each frame, and a
homography is fitted from those directly. This replaces cross-frame appearance
matching, which cannot work on this footage: grass carries almost no distinctive
texture (about 150 usable features per frame) and mowing stripes repeat, so
registration between frames breaks down after a few seconds.

Because every frame is solved independently, error does not accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
from ultralytics import YOLO

from pitchmap.calib.pitch import (
    CENTRE_CIRCLE_RADIUS_M,
    GOAL_AREA_DEPTH_M,
    GOAL_AREA_HALF_WIDTH_M,
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_HALF_WIDTH_M,
    PENALTY_SPOT_DISTANCE_M,
)
from pitchmap.config import PitchConfig

N_KEYPOINTS = 32
MIN_KEYPOINTS = 4


@dataclass(frozen=True)
class KeypointDetection:
    """Confident pitch keypoints in one frame, as index -> pixel position."""

    points: dict[int, tuple[float, float]]

    @property
    def count(self) -> int:
        return len(self.points)


def keypoint_positions(pitch: PitchConfig) -> dict[int, tuple[float, float]]:
    """Metric position of each of the 32 standard pitch keypoints.

    Index order follows the community-standard layout: the goal line nearest the
    origin first, then the halfway line, then the far goal line, then the two
    centre-circle side points. Regulation distances are used rather than the
    approximate ones some published configurations carry, so positions come out
    in true metres.
    """
    length, width = pitch.length_m, pitch.width_m
    half = width / 2.0
    box, box_w = PENALTY_AREA_DEPTH_M, PENALTY_AREA_HALF_WIDTH_M
    goal, goal_w = GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M
    spot, radius = PENALTY_SPOT_DISTANCE_M, CENTRE_CIRCLE_RADIUS_M

    return {
        0: (0.0, 0.0),
        1: (0.0, half - box_w),
        2: (0.0, half - goal_w),
        3: (0.0, half + goal_w),
        4: (0.0, half + box_w),
        5: (0.0, width),
        6: (goal, half - goal_w),
        7: (goal, half + goal_w),
        8: (spot, half),
        9: (box, half - box_w),
        10: (box, half - goal_w),
        11: (box, half + goal_w),
        12: (box, half + box_w),
        13: (length / 2.0, 0.0),
        14: (length / 2.0, half - radius),
        15: (length / 2.0, half + radius),
        16: (length / 2.0, width),
        17: (length - box, half - box_w),
        18: (length - box, half - goal_w),
        19: (length - box, half + goal_w),
        20: (length - box, half + box_w),
        21: (length - spot, half),
        22: (length - goal, half - goal_w),
        23: (length - goal, half + goal_w),
        24: (length, 0.0),
        25: (length, half - box_w),
        26: (length, half - goal_w),
        27: (length, half + goal_w),
        28: (length, half + box_w),
        29: (length, width),
        30: (length / 2.0 - radius, half),
        31: (length / 2.0 + radius, half),
    }


class PitchKeypointDetector:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        min_confidence: float = 0.5,
    ) -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.min_confidence = min_confidence

    def detect(self, frame: np.ndarray) -> KeypointDetection:
        result = self.model.predict(frame, verbose=False, device=self.device, conf=0.25)[0]
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return KeypointDetection({})

        keypoints = result.keypoints.data[0].cpu().numpy()
        return KeypointDetection(
            {
                index: (float(keypoints[index][0]), float(keypoints[index][1]))
                for index in range(min(N_KEYPOINTS, len(keypoints)))
                if keypoints[index][2] >= self.min_confidence
            }
        )

    def describe(self) -> str:
        return f"pitch-keypoints(min_conf={self.min_confidence}, device={self.device})"


def correspondences(
    detection: KeypointDetection,
    pitch: PitchConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Matched pixel and metric point arrays for a frame's keypoints."""
    layout = keypoint_positions(pitch)
    indices = sorted(index for index in detection.points if index in layout)
    image = np.array([detection.points[index] for index in indices], dtype=np.float64)
    world = np.array([layout[index] for index in indices], dtype=np.float64)
    return image, world
