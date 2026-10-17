"""Draw one heatmap per player across several runs, by the names in review.csv.

Track ids are numbered from scratch in every run, so id 5 in one clip and id 5
in the next are unrelated. What links clips is the player column a person fills
in each run's review.csv after checking its contact sheets. Positions of every
tracklet given the same name are pooled and drawn as one heatmap, so each clip
counts in proportion to the time the player was seen in it.

Tracklets judged 'mixed' are left out: they hold more than one person, and
pooling one would put someone else's positions into the heatmap.

Pool clips from one half at a time. Teams change ends at half time, so a
player's first and second halves would land on opposite sides of the pitch.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from pitchmap.config import load_config
from pitchmap.eval import review
from pitchmap.heat import accumulate, render
from pitchmap.io import artifacts


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs", nargs="+", required=True, help="Run directories to pool, e.g. every part of one half"
    )
    parser.add_argument("--out", required=True, help="Directory for the per-player heatmaps")


def named_positions(run_dir: Path) -> pd.DataFrame:
    """Valid positions of every named, non-mixed tracklet in one run."""
    paths = artifacts.run_paths(run_dir)
    name = paths.root.name
    if not paths.review.exists():
        print(f"{name}: no review.csv yet, skipped. Run 'review' and name the players first.")
        return pd.DataFrame()

    # Names are keyed to track ids, and refine renumbers those on every run.
    print(f"{name}: checking review.csv still matches the tracks")
    review.check_key(paths.review_key, artifacts.read_table(paths.tracks))

    judged = review.read_reviews(paths.review)
    named = judged[(judged["player"] != "") & (judged["verdict"] != "mixed")]
    if named.empty:
        print(f"{name}: no named tracklets, skipped.")
        return pd.DataFrame()

    positions = artifacts.read_table(paths.positions)
    meta = artifacts.read_meta(paths)
    fps = meta["video"]["fps"]
    # Clips from `cut` overlap; placing every frame on the recording's own frame
    # numbers lets the shared seconds be counted once. A run that is not a cut
    # clip keeps a source of its own, so it never collides with another.
    clip = meta.get("clip")
    source, first = (clip["source"], round(clip["start_s"] * fps)) if clip else (name, 0)
    pooled = positions[positions["valid"]].merge(named[["track_id", "player"]], on="track_id")
    return pooled.assign(
        run=name, seconds=1.0 / fps, source=source, abs_frame=pooled["frame"] + first
    )


def _warn_double_naming(pooled: pd.DataFrame) -> None:
    """A player on two tracklets in the same frame means one of them is someone else."""
    per_frame = pooled.groupby(["run", "key", "frame"])["track_id"].nunique()
    for run, key in per_frame[per_frame > 1].index.droplevel("frame").unique():
        rows = pooled[(pooled["run"] == run) & (pooled["key"] == key)]
        clashing = rows.groupby("frame")["track_id"].transform("nunique") > 1
        ids = ", ".join(str(t) for t in sorted(rows.loc[clashing, "track_id"].unique()))
        print(
            f"Warning: {rows['player'].iloc[0]} is on tracklets {ids} at the same time "
            f"in {run}; one of them is someone else."
        )


def _slug(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.casefold()).strip("_") or "player"


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    frames = [named_positions(Path(run_dir)) for run_dir in args.runs]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise SystemExit(
            "No named tracklets in any run. Fill the player column of each run's "
            "review.csv, then try again."
        )

    pooled = pd.concat(frames, ignore_index=True)
    # 'Saka' and 'saka ' are the same player; a typo in case should not split them.
    pooled = pooled.assign(key=pooled["player"].str.casefold())
    _warn_double_naming(pooled)
    before = len(pooled)
    pooled = pooled.drop_duplicates(subset=["key", "source", "abs_frame"])
    if len(pooled) < before:
        print(f"Counted {before - len(pooled)} overlapping frames once rather than twice.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("player_*.png"):
        stale.unlink()

    entries = []
    for _, rows in pooled.groupby("key"):
        player = rows["player"].iloc[0]
        seconds = float(rows["seconds"].sum())
        tracklets = len(rows[["run", "track_id"]].drop_duplicates())
        runs = rows["run"].nunique()
        grid = accumulate.occupancy(rows, config.pitch, config.heat)
        title = f"{player} - {seconds:.0f} s, {tracklets} tracklets in {runs} run(s)"
        render.render_single(grid, config.pitch, title, out / f"player_{_slug(player)}.png")
        entries.append((seconds, title, grid))

    entries.sort(key=lambda entry: entry[0], reverse=True)
    render.render_grid([(title, grid) for _, title, grid in entries], config.pitch, out / "all_players_grid.png")

    print(f"\n{len(entries)} players from {len(frames)} run(s):")
    for seconds, title, _ in entries:
        print(f"  {title}")
    print(f"Artifacts: {out}")
