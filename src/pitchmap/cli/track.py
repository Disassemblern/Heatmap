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
from pitchmap.viz.overlay import draw_hud, draw_tracks

WINDOW = "pitchmap track"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--tracker", choices=TRACKER_KINDS, default=None, help="Tracker backend")
    parser.add_argument("--conf", type=float, default=None, help="Override the tracker confidence")
    parser.add_argument(
        "--show", action="store_true", help="Display tracks in a window while running"
    )


def _detections_for_frame(group: pd.DataFrame) -> sv.Detections:
    return sv.Detections(
        xyxy=group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32),
        confidence=group["conf"].to_numpy(dtype=np.float32),
        class_id=group["cls"].to_numpy(dtype=int),
    )


def _display(frame, tracks, index: int) -> bool:
    """Draw tracks in a window. Returns False when the user asks to stop."""
    canvas = frame.copy()
    if tracks:
        boxes = np.array([[t.x1, t.y1, t.x2, t.y2] for t in tracks])
        canvas = draw_tracks(canvas, boxes, np.array([t.track_id for t in tracks]))
    canvas = draw_hud(canvas, [f"frame {index}  tracks {len(tracks)}"])
    cv2.imshow(WINDOW, canvas)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def run(args: argparse.Namespace) -> None:
    config = with_overrides(
        load_config(args.config), "track", kind=args.tracker, conf_min=args.conf
    )
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)
    detections = artifacts.read_table(paths.detections)
    by_frame = {frame: group for frame, group in detections.groupby("frame")}
    stop = int(detections["frame"].max()) + 1

    tracker = build_tracker(config.track)
    print(f"Tracking with {tracker.describe()}")
    if args.show:
        print("Showing tracks. Press q to stop early; work so far is still saved.")

    records = []
    for index, frame in tqdm(iter_frames(meta.path, stop=stop), total=stop, unit="frame"):
        group = by_frame.get(index)
        frame_detections = (
            _detections_for_frame(group) if group is not None else sv.Detections.empty()
        )
        frame_tracks = tracker.update(index, frame, frame_detections)

        if args.show and not _display(frame, frame_tracks, index):
            break

        records.extend(
            {
                "frame": index,
                "track_id": track.track_id,
                "x1": track.x1,
                "y1": track.y1,
                "x2": track.x2,
                "y2": track.y2,
                "conf": track.conf,
            }
            for track in frame_tracks
        )

    if args.show:
        cv2.destroyWindow(WINDOW)

    table = pd.DataFrame.from_records(records)
    artifacts.write_table(table, paths.tracks)
    mot.write_mot(table, paths.tracks_mot)

    lengths = table.groupby("track_id").size() / meta.fps
    long_tracks = int((lengths >= config.heat.min_track_s).sum())
    artifacts.record_stage(
        paths,
        "track",
        {
            "tracker": tracker.describe(),
            "tracks": int(table["track_id"].nunique()),
            "long_tracks": long_tracks,
        },
    )

    print(f"Wrote {len(table)} track rows over {int(table['track_id'].nunique())} track ids")
    print(
        f"Tracks lasting at least {config.heat.min_track_s:.0f} s: {long_tracks} "
        f"(median length {lengths.median():.1f} s)"
    )
    print(f"Artifacts: {paths.tracks}, {paths.tracks_mot}")
