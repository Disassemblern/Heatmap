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
    parser.add_argument(
        "--keep-officials",
        action="store_true",
        help="Also render tracklets the team stage labelled other (referees, goalkeepers)",
    )


def _drop_officials(paths, kept):
    """Remove tracklets the team stage labelled as belonging to neither team.

    Referees run where players run, so position cannot exclude them; only the
    jersey colour can. Tracklets labelled unknown are kept, since an unknown
    label means the evidence was too weak to act on.
    """
    if not paths.team.exists():
        return kept, []

    labels = artifacts.read_table(paths.team)
    other = set(labels.loc[labels["team"] == "other", "track_id"])
    dropped = [t for t in kept["track_id"] if t in other]
    return kept[~kept["track_id"].isin(other)], dropped


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

    if not args.keep_officials:
        kept, officials = _drop_officials(paths, kept)
        if officials:
            ids = ", ".join(str(int(t)) for t in officials)
            print(f"Skipped {len(officials)} official/other tracklet(s) by team label: {ids}")

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
        paths, "heatmap", {"tracklets": len(summary), "rendered": len(entries)}, config
    )
    print(f"Rendered {len(entries)} tracklet heatmaps plus an aggregate")
    print(f"Artifacts: {paths.heatmaps_dir}")
