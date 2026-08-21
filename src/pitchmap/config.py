"""Frozen configuration objects loaded from a YAML file."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


@dataclass(frozen=True)
class PitchConfig:
    length_m: float = 105.0
    width_m: float = 68.0


@dataclass(frozen=True)
class DetectConfig:
    model_path: str = "models/yolov8m.pt"
    conf_store: float = 0.10
    imgsz: int = 1280
    device: str = "cuda"


@dataclass(frozen=True)
class TrackConfig:
    kind: str = "deepsort"
    conf_min: float = 0.40
    max_age: int = 30
    n_init: int = 3


@dataclass(frozen=True)
class CalibConfig:
    keyframe_every_s: float = 10.0
    min_landmarks: int = 6
    ransac_reproj_px: float = 3.0
    drift_warn_m: float = 1.0
    drift_fail_m: float = 2.0


@dataclass(frozen=True)
class HeatConfig:
    cell_m: float = 0.5
    sigma_m: float = 2.0
    min_track_s: float = 10.0


@dataclass(frozen=True)
class Config:
    pitch: PitchConfig = PitchConfig()
    detect: DetectConfig = DetectConfig()
    track: TrackConfig = TrackConfig()
    calib: CalibConfig = CalibConfig()
    heat: HeatConfig = HeatConfig()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pitch": vars(self.pitch),
            "detect": vars(self.detect),
            "track": vars(self.track),
            "calib": vars(self.calib),
            "heat": vars(self.heat),
        }


def load_config(path: Path | str | None = None) -> Config:
    """Load configuration, falling back to dataclass defaults for absent keys."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return Config()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return Config(
        pitch=PitchConfig(**raw.get("pitch", {})),
        detect=DetectConfig(**raw.get("detect", {})),
        track=TrackConfig(**raw.get("track", {})),
        calib=CalibConfig(**raw.get("calib", {})),
        heat=HeatConfig(**raw.get("heat", {})),
    )


def with_overrides(config: Config, section: str, **overrides: Any) -> Config:
    """Return a copy of config with non-None overrides applied to one section."""
    applied = {key: value for key, value in overrides.items() if value is not None}
    if not applied:
        return config
    return replace(config, **{section: replace(getattr(config, section), **applied)})
