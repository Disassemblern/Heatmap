"""Project tracked boxes into pitch metres."""

from __future__ import annotations

import argparse

import numpy as np

from pitchmap.config import load_config
from pitchmap.io import artifacts
from pitchmap.project import ground


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument(
        "--smooth", action="store_true", help="Median-filter positions to damp foot-point jitter"
    )


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)

    tracks = artifacts.read_table(paths.tracks)
    calibration = artifacts.read_homographies(paths)

    positions = ground.project_tracks(
        tracks, calibration["matrices"], calibration["source"], config.pitch, meta.fps
    )
    if args.smooth:
        positions = ground.smooth_positions(positions)

    artifacts.write_table(positions, paths.positions)

    valid = positions["valid"]
    share = float(valid.mean()) * 100.0 if len(positions) else 0.0
    print(f"Projected {len(positions)} samples, {share:.1f} percent valid")

    counts = positions.loc[~valid, "invalid_reason"].value_counts()
    for reason, count in counts.items():
        print(f"  {reason}: {count}")

    speeds = ground._speeds(positions, valid.to_numpy(), meta.fps)
    moving = speeds[speeds > 0]
    if moving.size:
        print(
            f"Implied speed: median {np.median(moving):.1f} m/s, "
            f"99th percentile {np.percentile(moving, 99):.1f} m/s"
        )

    artifacts.record_stage(
        paths, "project", {"samples": len(positions), "valid_share": share, "smoothed": args.smooth}
    )
    print(f"Artifact: {paths.positions}")
