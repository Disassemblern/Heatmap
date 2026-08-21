"""Create the run directory for a clip."""

from __future__ import annotations

import argparse

from pitchmap.config import load_config
from pitchmap.io import artifacts
from pitchmap.io.video import read_meta


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, help="Path to the source clip")


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    meta = read_meta(args.video)
    paths = artifacts.create_run(args.video, meta, config.to_dict())

    print(f"Run directory: {paths.root}")
    print(
        f"Video: {meta.width}x{meta.height} at {meta.fps:.2f} fps, "
        f"{meta.n_frames} frames ({meta.duration_s:.1f} s)"
    )
