"""DeepSORT tracker backend.

Kept as the Part 1 baseline. Its Kalman motion model suits pedestrians better
than athletes, so it fragments identities during clustered play; replacing it is
a later milestone.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort

from pitchmap.config import TrackConfig
from pitchmap.track.base import Track


class DeepSortTracker:
    def __init__(self, config: TrackConfig) -> None:
        self.conf_min = config.conf_min
        self.tracker = DeepSort(max_age=config.max_age, n_init=config.n_init)

    def update(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> tuple[Track, ...]:
        inputs = [
            ([x1, y1, x2 - x1, y2 - y1], float(conf), int(class_id))
            for (x1, y1, x2, y2), conf, class_id in zip(
                detections.xyxy, detections.confidence, detections.class_id, strict=True
            )
            if conf >= self.conf_min
        ]

        tracks = self.tracker.update_tracks(inputs, frame=frame)
        return tuple(
            Track(
                track_id=int(track.track_id),
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                conf=float(track.det_conf if track.det_conf is not None else 0.0),
            )
            for track, bbox in ((t, t.to_tlbr()) for t in tracks if t.is_confirmed())
        )

    def describe(self) -> str:
        return f"deepsort(conf_min={self.conf_min})"
