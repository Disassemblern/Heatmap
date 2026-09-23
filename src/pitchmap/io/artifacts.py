"""Run directory layout and artifact read/write helpers.

A run directory holds everything derived from one clip. Stages read the
artifacts of earlier stages and write their own, so each stage is re-runnable in
isolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pitchmap.config import Config, changed_fields
from pitchmap.io.video import VideoMeta

RUNS_ROOT = Path("data/runs")


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def detections(self) -> Path:
        return self.root / "detections.parquet"

    @property
    def tracks(self) -> Path:
        return self.root / "tracks.parquet"

    @property
    def tracks_mot(self) -> Path:
        return self.root / "tracks_mot.txt"

    @property
    def keypoints(self) -> Path:
        return self.calib_dir / "keypoints.parquet"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings.npz"

    @property
    def tracks_raw(self) -> Path:
        return self.root / "tracks_raw.parquet"

    @property
    def team(self) -> Path:
        return self.root / "team.parquet"

    @property
    def team_features(self) -> Path:
        return self.root / "team_features.npz"

    @property
    def positions(self) -> Path:
        return self.root / "positions.parquet"

    @property
    def calib_dir(self) -> Path:
        return self.root / "calib"

    @property
    def keyframes(self) -> Path:
        return self.calib_dir / "keyframes.json"

    @property
    def homographies(self) -> Path:
        return self.calib_dir / "homographies.npz"

    @property
    def heatmaps_dir(self) -> Path:
        return self.root / "heatmaps"

    @property
    def overlays_dir(self) -> Path:
        return self.root / "overlays"

    @property
    def review_dir(self) -> Path:
        return self.root / "review"

    @property
    def review(self) -> Path:
        return self.root / "review.csv"

    @property
    def review_key(self) -> Path:
        return self.root / "review_key.json"

    @property
    def eval_report(self) -> Path:
        return self.root / "eval_report.json"


def run_paths(run_dir: Path | str) -> RunPaths:
    return RunPaths(root=Path(run_dir))


def create_run(video_path: Path | str, meta: VideoMeta) -> RunPaths:
    """Create (or reuse) the run directory for a clip and write its metadata.

    No config is recorded here. Stages run later, with their own flag
    overrides, so a snapshot taken at creation describes nothing that actually
    ran; record_stage stamps the settings each stage really used instead.
    """
    paths = run_paths(RUNS_ROOT / Path(video_path).stem)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.calib_dir.mkdir(exist_ok=True)

    existing = read_meta(paths) if paths.meta.exists() else {}
    write_meta(
        paths,
        {
            **existing,
            "video": vars(meta),
            "stages": existing.get("stages", {}),
        },
    )
    return paths


def read_meta(paths: RunPaths) -> dict[str, Any]:
    if not paths.meta.exists():
        raise SystemExit(
            f"No run at {paths.root} ({paths.meta.name} is missing). "
            "Check the clip name, or create the run with init-run first."
        )
    return json.loads(paths.meta.read_text(encoding="utf-8"))


def write_meta(paths: RunPaths, meta: dict[str, Any]) -> None:
    paths.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def video_meta(paths: RunPaths) -> VideoMeta:
    return VideoMeta(**read_meta(paths)["video"])


def record_stage(
    paths: RunPaths,
    stage: str,
    details: dict[str, Any] | None = None,
    config: Config | None = None,
) -> None:
    """Stamp a stage as completed, with optional details, for provenance.

    The recorded config holds only what differed from the defaults, so an empty
    dict means the stage ran on the checked-in configs/default.yaml.
    """
    meta = read_meta(paths)
    stages = dict(meta.get("stages", {}))
    stages[stage] = {
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(details or {}),
        **({"config": changed_fields(config)} if config is not None else {}),
    }
    write_meta(paths, {**meta, "stages": stages})


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}. Run the earlier stage first.")
    return pd.read_parquet(path)


def guard_truncation(path: Path, new_frames: int, force: bool) -> None:
    """Refuse to replace an artifact that already covers more frames.

    A --limit smoke test would otherwise leave the run half-populated and
    inconsistent with the stages that ran on the full clip.
    """
    if force or not path.exists():
        return

    try:
        existing = int(pd.read_parquet(path, columns=["frame"])["frame"].max()) + 1
    except (KeyError, ValueError, OSError):
        return

    if existing > new_frames:
        raise SystemExit(
            f"{path} already covers {existing} frames; this run would cover "
            f"{new_frames}. Re-run without --limit, pass --force to overwrite, or "
            f"use a separate run directory for the smoke test."
        )


def write_embeddings(
    paths: RunPaths,
    feat: np.ndarray,
    frame: np.ndarray,
    track_id: np.ndarray,
) -> None:
    """Persist per-track-row appearance embeddings, row-aligned with tracks.parquet.

    frame and track_id are stored redundantly so consumers can verify the
    alignment instead of trusting it.
    """
    np.savez(
        paths.embeddings,
        feat=feat.astype(np.float16),
        frame=frame.astype(np.int32),
        track_id=track_id.astype(np.int32),
    )


def read_embeddings(paths: RunPaths) -> dict[str, np.ndarray]:
    if not paths.embeddings.exists():
        raise FileNotFoundError(
            f"Missing artifact: {paths.embeddings}. Run 'track' with the deep_eiou backend first."
        )
    with np.load(paths.embeddings) as data:
        return {
            "feat": data["feat"].astype(np.float32),
            "frame": data["frame"],
            "track_id": data["track_id"],
        }


def write_homographies(
    paths: RunPaths,
    matrices: np.ndarray,
    source: np.ndarray,
    drift_m: np.ndarray,
    n_inliers: np.ndarray,
) -> None:
    paths.calib_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        paths.homographies,
        matrices=matrices,
        source=source,
        drift_m=drift_m,
        n_inliers=n_inliers,
    )


def read_homographies(paths: RunPaths) -> dict[str, np.ndarray]:
    if not paths.homographies.exists():
        raise FileNotFoundError(
            f"Missing artifact: {paths.homographies}. Run 'calibrate' first."
        )
    with np.load(paths.homographies) as data:
        return {key: data[key] for key in data.files}


def read_keyframes(paths: RunPaths) -> dict[int, dict[str, tuple[float, float]]]:
    """Return {frame_index: {landmark_name: (x_px, y_px)}}."""
    if not paths.keyframes.exists():
        return {}
    raw = json.loads(paths.keyframes.read_text(encoding="utf-8"))
    return {
        int(frame): {name: tuple(point) for name, point in landmarks.items()}
        for frame, landmarks in raw.items()
    }


def write_keyframes(
    paths: RunPaths,
    keyframes: dict[int, dict[str, tuple[float, float]]],
) -> None:
    paths.calib_dir.mkdir(parents=True, exist_ok=True)
    serialisable = {
        str(frame): {name: list(point) for name, point in landmarks.items()}
        for frame, landmarks in sorted(keyframes.items())
    }
    paths.keyframes.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
