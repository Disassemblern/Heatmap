"""Tracker construction by name."""

from __future__ import annotations

from pitchmap.config import Config
from pitchmap.track.base import Tracker
from pitchmap.track.bytetrack import ByteTracker
from pitchmap.track.deep_eiou import DeepEiouTracker

TRACKER_KINDS = ("deep_eiou", "bytetrack")


def build_tracker(config: Config, fps: float) -> Tracker:
    kind = config.track.kind
    if kind == "deep_eiou":
        return DeepEiouTracker(config.track, config.reid, fps)
    if kind == "bytetrack":
        return ByteTracker(config.track)
    raise ValueError(f"Unknown tracker kind: {kind}. Expected one of {TRACKER_KINDS}.")
