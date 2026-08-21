"""Render per-tracklet heatmaps.

Part 1 output is per tracklet, not per player: the current tracker fragments
identities, so one player may appear as several tracklets.
"""

from __future__ import annotations

import argparse

from pitchmap.config import load_config, with_overrides
from pitchmap.heat import accumulate, render
from pitchmap.io import artifacts


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument(
        "--min-seconds", type=float, default=None, help="Skip tracklets shorter than this"
    )
    parser.add_argument(
        "--top", type=int, default=12, help="How many tracklets to place on the contact sheet"
    )
    parser.add_argument(
        "--keep-off-pitch",
        action="store_true",
        help="Also render tracklets whose median position is off the pitch (coaches, staff)",
    )


def run(args: argparse.Namespace) -> None:
    config = with_overrides(load_config(args.config), "heat", min_track_s=args.min_seconds)
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)
    positions = artifacts.read_table(paths.positions)

    summary = accumulate.tracklet_summary(positions, meta.fps)
    kept = summary[summary["seconds"] >= config.heat.min_track_s]

    if not args.keep_off_pitch:
        on_pitch = accumulate.on_pitch_tracklets(positions, config.pitch)
        dropped = kept[~kept["track_id"].isin(on_pitch)]
        kept = kept[kept["track_id"].isin(on_pitch)]
        if len(dropped):
            ids = ", ".join(str(int(t)) for t in dropped["track_id"])
            print(f"Skipped {len(dropped)} tracklet(s) whose median position is off the pitch: {ids}")
    print(
        f"{len(summary)} tracklets total, {len(kept)} last at least "
        f"{config.heat.min_track_s:.0f} s"
    )

    entries = []
    for row in kept.itertuples():
        grid = accumulate.occupancy(
            positions[positions["track_id"] == row.track_id], config.pitch, config.heat
        )
        title = f"tracklet {row.track_id} - {row.seconds:.0f} s ({row.first_s:.0f}-{row.last_s:.0f} s)"
        render.render_single(
            grid, config.pitch, title, paths.heatmaps_dir / f"player_{row.track_id}.png"
        )
        entries.append((title, grid))

    if entries:
        render.render_grid(
            entries[: args.top], config.pitch, paths.heatmaps_dir / "all_tracklets_grid.png"
        )

    render.render_single(
        accumulate.occupancy(positions, config.pitch, config.heat),
        config.pitch,
        "all tracked positions",
        paths.heatmaps_dir / "aggregate.png",
    )

    artifacts.record_stage(
        paths, "heatmap", {"tracklets": len(summary), "rendered": len(entries)}
    )
    print(f"Rendered {len(entries)} tracklet heatmaps plus an aggregate")
    print(f"Artifacts: {paths.heatmaps_dir}")
