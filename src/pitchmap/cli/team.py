"""Cluster tracklets into two teams by jersey colour.

Also identifies the tracklets that belong to neither team (officials,
goalkeepers, staff), which the heatmap stage excludes by default and the refine
stage uses as a hard constraint on merging.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from pitchmap.config import load_config, with_overrides
from pitchmap.heat import accumulate
from pitchmap.io import artifacts
from pitchmap.refine import geometry
from pitchmap.team import assign, jersey


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument(
        "--samples", type=int, default=None, help="Crops sampled per tracklet"
    )
    parser.add_argument(
        "--other-ratio",
        type=float,
        default=None,
        help="Distance ratio beyond which a tracklet belongs to neither team",
    )


def _fit_ids(paths: artifacts.RunPaths, tracks, fps: float, config) -> set[int]:
    """Long, on-pitch tracklets: the ones worth fitting the two clusters on."""
    lengths_s = tracks.groupby("track_id").size() / fps
    long_ids = set(lengths_s[lengths_s >= config.team.min_cluster_track_s].index)

    if not paths.positions.exists():
        return long_ids

    positions = artifacts.read_table(paths.positions)
    # positions may have been produced from refined tracks, whose ids are a
    # different numbering to the table being read here. Only apply the on-pitch
    # filter when the two id spaces actually correspond.
    shared = set(positions["track_id"]) & set(tracks["track_id"])
    if len(shared) < 0.8 * len(set(tracks["track_id"])):
        print("Skipping the on-pitch filter: positions.parquet uses different track ids.")
        return long_ids

    return long_ids & accumulate.on_pitch_tracklets(positions, config.pitch)


def _goalkeepers(tracks, detections) -> set[int]:
    """Tracklets the detector mostly classed as goalkeeper.

    A goalkeeper's kit is chosen to differ from both teams. Fitting the two team
    colours with a goalkeeper included drags one centroid onto that kit and
    lumps both outfield teams together, which is what happened on short clips,
    where a goalkeeper's long, steady tracklet carries a lot of weight. Track
    boxes are the detector's own boxes, so a join on frame and box recovers each
    row's class.
    """
    if "cls" not in detections.columns:
        return set()
    key = ["frame", "x1", "y1", "x2", "y2"]

    def rounded(table):
        return table.assign(**{column: table[column].round(2) for column in key[1:]})

    joined = rounded(tracks[["track_id", *key]]).merge(
        rounded(detections[[*key, "cls"]]).drop_duplicates(key), on=key, how="left"
    )
    share = (joined["cls"] == "goalkeeper").groupby(joined["track_id"]).mean()
    return set(share[share > 0.5].index)


def _alone_only(features: jersey.JerseyFeatures, tracks) -> jersey.JerseyFeatures:
    """Keep the colour only of crops whose box overlaps no other player's.

    Where two boxes overlap, either torso crop may show the other player's
    shirt: a navy player standing behind a red one reads as red. Every later
    use of colour (team labels, the team colour fit, the colour split in refine)
    skips crops without pixels, so blanking them here keeps all of them honest.
    """
    touching = geometry.overlapping_rows(tracks)[features.row_index]
    return replace(features, n_pixels=np.where(touching, 0, features.n_pixels))


def run(args: argparse.Namespace) -> None:
    config = with_overrides(
        load_config(args.config),
        "team",
        samples_per_tracklet=args.samples,
        other_dist_ratio=args.other_ratio,
    )
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)

    source = paths.tracks_raw if paths.tracks_raw.exists() else paths.tracks
    tracks = artifacts.read_table(source)
    print(f"Reading tracks from {source.name} ({tracks['track_id'].nunique()} tracklets)")

    sampled = jersey.sample_rows(tracks, config.team.samples_per_tracklet)
    print(f"Sampling {len(sampled)} crops for jersey colour")
    features = _alone_only(jersey.extract_features(meta.path, sampled), tracks)
    print(f"Kept the colour of {int((features.n_pixels > 0).sum())} crops where the player stood alone")

    per_tracklet, colours = assign.tracklet_features(tracks, features)
    if per_tracklet.empty:
        raise SystemExit("No usable jersey colours found; cannot assign teams.")

    goalkeepers = _goalkeepers(tracks, artifacts.read_table(paths.detections))
    fit_ids = _fit_ids(paths, tracks, meta.fps, config) - goalkeepers
    if goalkeepers:
        print(f"Leaving {len(goalkeepers)} goalkeeper tracklet(s) out of the team colour fit")
    labels, centroids, space = assign.assign_teams(
        per_tracklet, colours, fit_ids, config.team
    )

    np.savez(
        paths.team_features,
        row_index=features.row_index,
        lab=features.lab,
        n_pixels=features.n_pixels,
        centroids=centroids.astype(np.float32),
        space=np.str_(space),
        source_n_rows=np.int64(len(tracks)),
    )
    artifacts.write_table(labels, paths.team)

    counts = labels["team"].value_counts()
    artifacts.record_stage(
        paths,
        "team",
        {
            "tracklets": int(len(labels)),
            "fitted_on": len(fit_ids),
            "goalkeepers_excluded": len(goalkeepers),
            **{f"n_{name}": int(count) for name, count in counts.items()},
        },
        config,
    )

    print(f"Fitted two clusters on {len(fit_ids)} long on-pitch tracklets")
    for name, count in counts.items():
        print(f"  {name:8s} {count}")
    print(f"Artifacts: {paths.team}, {paths.team_features}")
