"""Score tracking identity against human judgements."""

from __future__ import annotations

import argparse
import json

from pitchmap.config import load_config
from pitchmap.eval import review
from pitchmap.io import artifacts


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)

    if not paths.review.exists():
        raise SystemExit(
            f"Missing {paths.review}. Run 'review' first, then judge the contact sheets."
        )

    tracks = artifacts.read_table(paths.tracks)
    review.check_key(paths.review_key, tracks)
    lengths_s = tracks.groupby("track_id").size() / meta.fps

    predicted = None
    if paths.team.exists():
        team = artifacts.read_table(paths.team)
        predicted = team.set_index("track_id")["team"]

    report = review.evaluate(review.read_reviews(paths.review), lengths_s, predicted)
    print(report.format())

    paths.eval_report.write_text(json.dumps(vars(report), indent=2), encoding="utf-8")
    artifacts.record_stage(paths, "eval", vars(report), config)
    print(f"\nArtifact: {paths.eval_report}")
