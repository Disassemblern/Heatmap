"""Frozen configuration objects loaded from a YAML file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    kind: str = "deep_eiou"
    # bytetrack only
    conf_min: float = 0.40
    max_age: int = 30
    n_init: int = 3
    # deep_eiou (upstream demo defaults; tuning starting points)
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 60
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    min_box_area: float = 10.0
    # Camera motion compensation. The vendored tracker has multi_gmc, but nothing
    # reads this flag yet.
    cmc: bool = False


@dataclass(frozen=True)
class ReidConfig:
    model_path: str = "models/osnet_sports.pth.tar-60"
    num_classes: int = 616
    batch: int = 64
    fp16: bool = True
    device: str = "cuda"


@dataclass(frozen=True)
class RefineConfig:
    use_split: bool = True
    use_connect: bool = True
    # DBSCAN neighbourhood for detecting an identity switch inside a tracklet.
    # Upstream ships 0.7; measured on this footage, 0.4 raises tracklet purity
    # from 0.79 to 0.92, while tighter values split on noise.
    eps: float = 0.4
    min_samples: int = 10
    max_k: int = 3
    min_len_frames: int = 100
    spatial_factor: float = 1.0
    merge_dist_thres: float = 0.4
    use_team_constraint: bool = True
    # A team label only vetoes a merge when it is at least this confident; a
    # weak label blocking a correct merge is worse than no label at all.
    min_veto_confidence: float = 0.35
    # Reject a merge that would require the player to run faster than this
    # between the two fragments. Measured in pitch metres, so it needs
    # calibration; without it the check is skipped.
    use_speed_constraint: bool = True
    max_speed_ms: float = 12.0


@dataclass(frozen=True)
class TeamConfig:
    samples_per_tracklet: int = 80
    min_rows: int = 5
    min_cluster_track_s: float = 10.0
    # Distance to the nearer team centroid, as a fraction of the gap between
    # the two centroids, beyond which a tracklet belongs to neither team.
    # Calibrated on the test clip: real players reach 0.48, officials 0.73.
    other_dist_ratio: float = 0.55
    min_silhouette: float = 0.25


@dataclass(frozen=True)
class CalibConfig:
    keyframe_every_s: float = 10.0
    ransac_reproj_px: float = 3.0
    drift_warn_m: float = 1.0


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
    reid: ReidConfig = ReidConfig()
    refine: RefineConfig = RefineConfig()
    team: TeamConfig = TeamConfig()
    calib: CalibConfig = CalibConfig()
    heat: HeatConfig = HeatConfig()



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
        reid=ReidConfig(**raw.get("reid", {})),
        refine=RefineConfig(**raw.get("refine", {})),
        team=TeamConfig(**raw.get("team", {})),
        calib=CalibConfig(**raw.get("calib", {})),
        heat=HeatConfig(**raw.get("heat", {})),
    )


def with_overrides(config: Config, section: str, **overrides: Any) -> Config:
    """Return a copy of config with non-None overrides applied to one section."""
    applied = {key: value for key, value in overrides.items() if value is not None}
    if not applied:
        return config
    return replace(config, **{section: replace(getattr(config, section), **applied)})


def changed_fields(config: Config) -> dict[str, dict[str, Any]]:
    """Settings that differ from the dataclass defaults, by section.

    Stamped into run metadata per stage. The full defaults are in
    configs/default.yaml under version control, so only the deltas carry
    information, and a stage's own flag overrides show up here.
    """
    defaults = asdict(Config())
    return {
        section: {
            key: value
            for key, value in values.items()
            if defaults[section][key] != value
        }
        for section, values in asdict(config).items()
        if values != defaults[section]
    }
