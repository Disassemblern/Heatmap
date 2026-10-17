"""Assign track ids to stored detections."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm

from pitchmap.config import load_config, with_overrides
from pitchmap.io import artifacts, mot
from pitchmap.io.video import iter_frames
from pitchmap.track.factory import TRACKER_KINDS, build_tracker
from pitchmap.track.metrics import format_summary, fragmentation_summary
from pitchmap.viz.overlay import draw_hud, draw_tracks, show

WINDOW = "pitchmap track"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--tracker", choices=TRACKER_KINDS, default=None, help="Tracker backend")
    parser.add_argument("--conf", type=float, default=None, help="Override the tracker confidence")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Detected classes to track (default: player goalkeeper)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display tracks in a window while running"
    )


def _detections_for_frame(group: pd.DataFrame) -> sv.Detections:
    return sv.Detections(
        xyxy=group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32),
        confidence=group["conf"].to_numpy(dtype=np.float32),
    )


def _one_box_per_person(detections: pd.DataFrame, iou: float) -> pd.DataFrame:
    """Keep the most confident of overlapping boxes, whatever class each was given.

    The detector suppresses duplicates only within a class, so one goalkeeper can
    carry a 'goalkeeper' and a 'player' box, and the tracker then follows him
    with two ids. Done before the class filter, so a 'player' box sitting under a
    more confident 'referee' box, which is an official, is dropped with it.
    """
    if iou <= 0:
        return detections
    detections = detections.reset_index(drop=True)
    scored = detections[["x1", "y1", "x2", "y2", "conf"]].to_numpy(dtype=float)
    keep = np.zeros(len(detections), dtype=bool)
    for rows in detections.groupby("frame").indices.values():
        keep[rows] = sv.box_non_max_suppression(scored[rows], iou_threshold=iou)
    print(f"Dropped {int((~keep).sum())} boxes duplicating a more confident one on the same person")
    return detections[keep]


def _keep_classes(detections: pd.DataFrame, classes: tuple[str, ...]) -> pd.DataFrame:
    """Drop the detected classes that are not worth tracking.

    Referees and the ball are detected and stored, but a referee tracked like a
    player produces the longest, cleanest tracklet in the clip and is not a
    player at all. Filtering here rather than after tracking keeps the vendored
    tracker untouched, which does not carry a class through anyway.
    """
    kept = detections[detections["cls"].isin(classes)]
    if kept.empty:
        raise SystemExit(
            f"No detections left after filtering to {', '.join(classes)}. "
            f"detections.parquet holds classes {sorted(detections['cls'].unique())[:6]}, "
            "which means it predates the football detector. Re-run detect first."
        )

    dropped = detections["cls"].value_counts().drop(labels=list(classes), errors="ignore")
    if not dropped.empty:
        summary = ", ".join(f"{name} {count}" for name, count in dropped.items())
        print(f"Tracking {', '.join(classes)}; skipped {summary}")
    return kept


def _display(frame, tracks, index: int) -> bool:
    """Draw tracks in a window. Returns False when the user asks to stop."""
    canvas = frame.copy()
    if tracks:
        boxes = np.array([[t.x1, t.y1, t.x2, t.y2] for t in tracks])
        canvas = draw_tracks(canvas, boxes, np.array([t.track_id for t in tracks]))
    canvas = draw_hud(canvas, [f"frame {index}  tracks {len(tracks)}"])
    return show(WINDOW, canvas)


def _remove_stale_downstream(paths: artifacts.RunPaths) -> None:
    """New tracks invalidate everything the refine and team stages produced."""
    stale = [
        path
        for path in (paths.tracks_raw, paths.team, paths.team_features)
        if path.exists()
    ]
    for path in stale:
        path.unlink()
    if stale:
        names = ", ".join(path.name for path in stale)
        print(f"Removed stale downstream artifacts ({names}); re-run team and refine.")


def run(args: argparse.Namespace) -> None:
    config = with_overrides(
        load_config(args.config),
        "track",
        kind=args.tracker,
        conf_min=args.conf,
        classes=args.classes,
    )
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)
    detections = artifacts.read_table(paths.detections)
    stop = int(detections["frame"].max()) + 1
    detections = _one_box_per_person(detections, config.track.cross_class_nms_iou)
    detections = _keep_classes(detections, config.track.classes)
    by_frame = {frame: group for frame, group in detections.groupby("frame")}

    tracker = build_tracker(config, meta.fps)
    print(f"Tracking with {tracker.describe()}")
    if args.show:
        print("Showing tracks. Press q to stop early; work so far is still saved.")

    records = []
    feats: list[np.ndarray | None] = []
    for index, frame in tqdm(iter_frames(meta.path, stop=stop), total=stop, unit="frame"):
        group = by_frame.get(index)
        frame_detections = (
            _detections_for_frame(group) if group is not None else sv.Detections.empty()
        )
        frame_tracks = tracker.update(index, frame, frame_detections)

        if args.show and not _display(frame, frame_tracks, index):
            break

        for track in frame_tracks:
            records.append(
                {
                    "frame": index,
                    "track_id": track.track_id,
                    "x1": track.x1,
                    "y1": track.y1,
                    "x2": track.x2,
                    "y2": track.y2,
                    "conf": track.conf,
                }
            )
            feats.append(track.feat)

    if args.show:
        cv2.destroyWindow(WINDOW)

    table = pd.DataFrame.from_records(records)
    artifacts.write_table(table, paths.tracks)
    mot.write_mot(table, paths.tracks_mot)
    _remove_stale_downstream(paths)
    _save_embeddings(paths, table, feats)

    summary = fragmentation_summary(table, meta.fps, stop)
    artifacts.record_stage(
        paths, "track", {"tracker": tracker.describe(), **summary}, config
    )

    print(format_summary(summary))
    print(f"Artifacts: {paths.tracks}, {paths.tracks_mot}")


def _save_embeddings(
    paths: artifacts.RunPaths,
    table: pd.DataFrame,
    feats: list[np.ndarray | None],
) -> None:
    if not feats or any(feat is None for feat in feats):
        if paths.embeddings.exists():
            paths.embeddings.unlink()
            print("Removed stale embeddings.npz (backend produced no embeddings).")
        return

    artifacts.write_embeddings(
        paths,
        np.stack(feats),
        table["frame"].to_numpy(),
        table["track_id"].to_numpy(),
    )
    print(f"Cached {len(feats)} appearance embeddings: {paths.embeddings}")
