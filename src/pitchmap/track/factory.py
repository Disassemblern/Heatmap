"""Tracker construction by name."""

from __future__ import annotations

from pitchmap.config import TrackConfig
from pitchmap.track.base import Tracker
from pitchmap.track.bytetrack import ByteTracker
from pitchmap.track.deepsort import DeepSortTracker

TRACKER_KINDS = ("deepsort", "bytetrack")


def build_tracker(config: TrackConfig) -> Tracker:
    if config.kind == "deepsort":
        return DeepSortTracker(config)
    if config.kind == "bytetrack":
        return ByteTracker(config)
    raise ValueError(f"Unknown tracker kind: {config.kind}. Expected one of {TRACKER_KINDS}.")
