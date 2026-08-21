"""ByteTrack backend via supervision.

Motion-only association with a low-confidence recovery pass, so it consumes the
full stored detection set rather than a pre-filtered one.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from pitchmap.config import TrackConfig
from pitchmap.track.base import Track


class ByteTracker:
    def __init__(self, config: TrackConfig) -> None:
        self.conf_min = config.conf_min
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.conf_min,
            lost_track_buffer=config.max_age,
            minimum_consecutive_frames=config.n_init,
        )

    def update(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> tuple[Track, ...]:
        tracked = self.tracker.update_with_detections(detections)
        if tracked.tracker_id is None:
            return ()

        return tuple(
            Track(
                track_id=int(track_id),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                conf=float(conf),
            )
            for (x1, y1, x2, y2), track_id, conf in zip(
                tracked.xyxy, tracked.tracker_id, tracked.confidence, strict=True
            )
        )

    def describe(self) -> str:
        return f"bytetrack(conf_min={self.conf_min})"
