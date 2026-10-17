"""Create the run directory for a clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitchmap.config import load_config
from pitchmap.io import artifacts
from pitchmap.io.video import read_meta


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, help="Path to the source clip")


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    meta = read_meta(args.video)
    paths = artifacts.create_run(args.video, meta)

    # A clip made by `cut` carries its exact start in the recording; `link` and
    # `combine` need it to line frames up across overlapping clips.
    sidecar = Path(args.video).with_suffix(".json")
    if sidecar.exists():
        clip = json.loads(sidecar.read_text(encoding="utf-8"))
        artifacts.write_meta(paths, {**artifacts.read_meta(paths), "clip": clip})
        print(f"Clip of {clip['source']} starting at {clip['start_s']:.2f} s")

    print(f"Run directory: {paths.root}")
    print(
        f"Video: {meta.width}x{meta.height} at {meta.fps:.2f} fps, "
        f"{meta.n_frames} frames ({meta.duration_s:.1f} s)"
    )
