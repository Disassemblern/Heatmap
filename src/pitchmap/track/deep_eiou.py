"""Deep-EIoU tracker backend with OSNet appearance embeddings.

Sports-specific association: ExpansionIoU instead of Kalman-predicted IoU, plus
appearance matching against a sports-trained ReID model. Vendored tracker; see
src/pitchmap/vendor/README.md.

Unlike the other backends this one must receive the FULL stored detection set
(the 0.10 confidence floor): its dual-threshold logic associates low-confidence
detections against existing tracks instead of discarding them.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import supervision as sv

from pitchmap.config import ReidConfig, TrackConfig
from pitchmap.reid.embedder import EMBEDDING_DIM, ReidEmbedder
from pitchmap.track.base import Track
from pitchmap.vendor.deep_eiou import Deep_EIoU


class DeepEiouTracker:
    def __init__(self, config: TrackConfig, reid: ReidConfig, fps: float) -> None:
        self.config = config
        self.embedder = ReidEmbedder(reid)
        # The vendored tracker reads exactly these fields from its args object.
        self.tracker = Deep_EIoU(
            Namespace(
                track_high_thresh=config.track_high_thresh,
                track_low_thresh=config.track_low_thresh,
                new_track_thresh=config.new_track_thresh,
                track_buffer=config.track_buffer,
                match_thresh=config.match_thresh,
                proximity_thresh=config.proximity_thresh,
                appearance_thresh=config.appearance_thresh,
                with_reid=True,
            ),
            frame_rate=fps,
        )

    def update(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> tuple[Track, ...]:
        if len(detections.xyxy) == 0:
            outputs = np.zeros((0, 5), dtype=np.float32)
            embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        else:
            outputs = np.hstack(
                [detections.xyxy, detections.confidence[:, None]]
            ).astype(np.float32)
            embeddings = self.embedder.embed(frame, outputs[:, :4])

        online = self.tracker.update(outputs, embeddings)

        tracks = []
        for strack in online:
            # last_tlbr is the raw matched detection box; the Kalman-smoothed
            # state would shift foot points that projection depends on.
            x1, y1, x2, y2 = strack.last_tlbr
            if (x2 - x1) * (y2 - y1) <= self.config.min_box_area:
                continue
            tracks.append(
                Track(
                    track_id=int(strack.track_id),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    conf=float(strack.score),
                    feat=None if strack.curr_feat is None else strack.curr_feat.astype(np.float32),
                )
            )
        return tuple(tracks)

    def describe(self) -> str:
        return (
            f"deep_eiou(high={self.config.track_high_thresh}, "
            f"low={self.config.track_low_thresh}, buffer={self.config.track_buffer}, "
            f"appearance={self.config.appearance_thresh}) + {self.embedder.describe()}"
        )
