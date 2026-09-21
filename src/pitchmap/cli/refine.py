"""Offline tracklet refinement: split hidden identity switches, reconnect fragments.

Always refines from tracks_raw.parquet (created from tracks.parquet on first
run), so re-running with different parameters is idempotent. The refined output
replaces tracks.parquet and tracks_mot.txt, so every downstream stage runs
unchanged.
"""

from __future__ import annotations

import argparse
import shutil

import numpy as np

from pitchmap.config import load_config, with_overrides
from pitchmap.io import artifacts, mot
from pitchmap.refine import convert, drivers, geometry
from pitchmap.team import assign, jersey
from pitchmap.track.metrics import format_summary, fragmentation_summary


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--no-split", action="store_true", help="Skip the identity-switch split")
    parser.add_argument("--no-connect", action="store_true", help="Skip the fragment reconnect")
    parser.add_argument(
        "--no-team-constraint", action="store_true", help="Merge without the team veto"
    )
    parser.add_argument("--eps", type=float, default=None, help="Override the split DBSCAN eps")
    parser.add_argument(
        "--merge-dist", type=float, default=None, help="Override the merge distance threshold"
    )


def _load_raw(paths: artifacts.RunPaths):
    """The raw (pre-refine) tracks table, creating the backup on first run."""
    if not paths.tracks_raw.exists():
        if not paths.tracks.exists():
            raise SystemExit(f"Missing artifact: {paths.tracks}. Run 'track' first.")
        shutil.copyfile(paths.tracks, paths.tracks_raw)
        print(f"Backed up raw tracks to {paths.tracks_raw}")
    return artifacts.read_table(paths.tracks_raw)


def _load_aligned_embeddings(paths: artifacts.RunPaths, raw) -> np.ndarray:
    data = artifacts.read_embeddings(paths)
    if len(data["feat"]) != len(raw) or not (
        np.array_equal(data["frame"], raw["frame"].to_numpy())
        and np.array_equal(data["track_id"], raw["track_id"].to_numpy())
    ):
        raise SystemExit(
            "embeddings.npz is not aligned with tracks_raw.parquet. "
            "Re-run 'track' with the deep_eiou backend."
        )
    return data["feat"]


def _pitch_positions(paths: artifacts.RunPaths, raw, refine):
    """Foot points of the raw rows in pitch metres, for the speed constraint."""
    if not refine.use_speed_constraint:
        return None
    if not paths.homographies.exists():
        print("No calibration found; merging without the speed constraint.")
        return None

    calibration = artifacts.read_homographies(paths)
    return geometry.pitch_positions(raw, calibration["matrices"], calibration["source"])


def _load_jersey_cache(paths: artifacts.RunPaths):
    """Cached jersey colours and team centroids, when the team stage has run."""
    if not paths.team_features.exists():
        return None
    with np.load(paths.team_features) as data:
        return (
            jersey.JerseyFeatures(
                row_index=data["row_index"], lab=data["lab"], n_pixels=data["n_pixels"]
            ),
            data["centroids"].astype(np.float64),
            str(data["space"]) if "space" in data else assign.CHROMA,
        )


def _labels_for(tracklet_set, cache, config) -> drivers.TeamLabels:
    """Derive team labels for whatever grouping the tracklets currently have.

    Labels are always computed from the cached per-detection colours rather than
    read from team.parquet, because splitting and merging change tracklet
    membership and renumber ids: any stored label set is keyed to a grouping
    that no longer exists.
    """
    features, centroids, space = cache
    table = assign.label_rows(tracklet_set.rows, features, centroids, space, config.team)
    return {
        int(row.track_id): (str(row.team), float(row.confidence))
        for row in table.itertuples()
    }


def _rewrite_team_labels(
    paths: artifacts.RunPaths,
    config,
    tracklet_set: convert.TrackletSet,
    raw,
    cache,
) -> None:
    """Re-key team labels onto the refined track ids.

    Refinement renumbers tracklets and changes which rows belong together, so
    the labels the team stage produced no longer apply. Labels are re-derived
    for the final grouping from the cached jersey colours, against the same
    centroids, rather than inherited.
    """
    if cache is None:
        return
    features, centroids, space = cache

    # convert.to_table renumbers by first frame; mirror that ordering here.
    ordered = sorted(
        tracklet_set.rows.items(), key=lambda item: min(raw["frame"].iloc[item[1]])
    )
    rows_by_new_id = {new_id: rows for new_id, (_, rows) in enumerate(ordered, start=1)}

    labels = assign.label_rows(rows_by_new_id, features, centroids, space, config.team)
    artifacts.write_table(labels, paths.team)

    counts = labels["team"].value_counts().to_dict()
    print(f"Re-keyed team labels to refined ids: {counts}")


def run(args: argparse.Namespace) -> None:
    config = with_overrides(
        load_config(args.config),
        "refine",
        use_split=False if args.no_split else None,
        use_connect=False if args.no_connect else None,
        use_team_constraint=False if args.no_team_constraint else None,
        eps=args.eps,
        merge_dist_thres=args.merge_dist,
    )
    refine = config.refine
    if not refine.use_split and not refine.use_connect:
        raise SystemExit("Nothing to do: both split and connect are disabled.")

    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)
    raw = _load_raw(paths)
    feats = _load_aligned_embeddings(paths, raw)

    tracklet_set = convert.to_tracklets(raw, feats)
    n_input = len(tracklet_set.tracklets)
    print(f"Refining {n_input} tracklets from {paths.tracks_raw.name}")

    if refine.use_split:
        tracklet_set = drivers.split_pass(tracklet_set, refine)
        print(f"After split: {len(tracklet_set.tracklets)} tracklets")
    n_split = len(tracklet_set.tracklets)

    positions = _pitch_positions(paths, raw, refine)

    cache = _load_jersey_cache(paths)
    rejected: dict[str, int] = {}
    if refine.use_connect:
        labels = None
        if refine.use_team_constraint:
            if cache is None:
                print("No jersey colours cached (run 'team' first); merging without the veto.")
            else:
                # Derived after splitting, so split products are labelled on
                # their own rows rather than inheriting a label that described
                # the mixed tracklet they came from.
                labels = _labels_for(tracklet_set, cache, config)
        tracklet_set, rejected = drivers.merge_pass(
            tracklet_set, refine, labels, positions, meta.fps
        )
        blocked = ", ".join(f"{count} by {name}" for name, count in rejected.items() if count)
        print(f"After merge: {len(tracklet_set.tracklets)} tracklets ({blocked or 'nothing blocked'})")

    table = convert.to_table(tracklet_set, raw)
    artifacts.write_table(table, paths.tracks)
    mot.write_mot(table, paths.tracks_mot)
    _rewrite_team_labels(paths, config, tracklet_set, raw, cache)

    n_frames = int(raw["frame"].max()) + 1
    summary = fragmentation_summary(table, meta.fps, n_frames)
    artifacts.record_stage(
        paths,
        "refine",
        {
            "input_tracklets": n_input,
            "after_split": n_split,
            "after_merge": len(tracklet_set.tracklets),
            "rejected_team": rejected.get("team", 0),
            "rejected_speed": rejected.get("speed", 0),
            "rejected_spatial": rejected.get("spatial", 0),
            "use_split": refine.use_split,
            "use_connect": refine.use_connect,
            "use_team_constraint": refine.use_team_constraint,
            "eps": refine.eps,
            "merge_dist_thres": refine.merge_dist_thres,
            **summary,
        },
        config,
    )

    print(format_summary(summary))
    print(f"Artifacts: {paths.tracks}, {paths.tracks_mot}")
