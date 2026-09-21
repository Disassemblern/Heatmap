"""Render tracklet contact sheets and a template for judging them.

Annotation cost is what limits how much ground truth exists, so this stage does
the mechanical part: it renders each tracklet as a strip of crops spanning its
life and writes a CSV pre-filled with the pipeline's own guesses. Judging is
then reading a few images and correcting a column.
"""

from __future__ import annotations

import argparse

import pandas as pd

from pitchmap.config import load_config
from pitchmap.eval import review, sheets
from pitchmap.io import artifacts


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument(
        "--min-seconds", type=float, default=20.0, help="Only review tracklets this long"
    )
    parser.add_argument("--crops", type=int, default=6, help="Crops per tracklet")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing review CSV"
    )


def run(args: argparse.Namespace) -> None:
    load_config(args.config)
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)
    tracks = artifacts.read_table(paths.tracks)

    lengths_s = tracks.groupby("track_id").size() / meta.fps
    track_ids = sorted(lengths_s[lengths_s >= args.min_seconds].index)
    print(f"{len(track_ids)} tracklets last at least {args.min_seconds:.0f} s")

    predicted = _predicted_teams(paths)
    labels = {
        track_id: f"{lengths_s[track_id]:.0f}s {predicted.get(track_id, '')}".strip()
        for track_id in track_ids
    }

    wanted = sheets.sample_crops(tracks, track_ids, args.crops)
    written = sheets.render(meta.path, tracks, wanted, labels, paths.review_dir)
    print(f"Wrote {len(written)} contact sheet(s) to {paths.review_dir}")

    if paths.review.exists() and not args.overwrite:
        print(f"Keeping the existing {paths.review.name}; pass --overwrite to reset it.")
    else:
        template = pd.DataFrame(
            {
                "track_id": track_ids,
                "verdict": review.VERDICT_CLEAN,
                "team": [predicted.get(track_id, "") for track_id in track_ids],
                "player": "",
            }
        )
        template.to_csv(paths.review, index=False)
        print(f"Wrote judgement template to {paths.review}")
    review.write_key(paths.review_key, tracks)

    print(
        "\nOpen the sheets, then edit the CSV: set verdict to mixed where a strip\n"
        "shows more than one person, correct the team column, and optionally name\n"
        "the player. Then run: python -m pitchmap eval --run " + str(paths.root)
    )


def _predicted_teams(paths: artifacts.RunPaths) -> dict[int, str]:
    if not paths.team.exists():
        return {}
    table = artifacts.read_table(paths.team)
    return {int(row.track_id): str(row.team) for row in table.itertuples()}
