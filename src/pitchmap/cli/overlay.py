"""Render calibration or tracking sanity videos."""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from pitchmap.config import load_config
from pitchmap.io import artifacts
from pitchmap.io.video import iter_frames
from pitchmap.viz import overlay


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--kind", choices=("calib", "track"), default="calib", help="Which video")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N frames")


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)
    stop = min(meta.n_frames, args.limit) if args.limit else meta.n_frames

    if args.kind == "calib":
        _render_calibration(paths, meta, config, stop)
    else:
        _render_tracking(paths, meta, config, stop)


def _render_calibration(paths, meta, config, stop: int) -> None:
    calibration = artifacts.read_homographies(paths)
    matrices, source, drift = (
        calibration["matrices"], calibration["source"], calibration["drift_m"]
    )
    output = paths.overlays_dir / "calibration_check.mp4"
    video = overlay.writer(output, meta.fps, meta.width, meta.height)

    try:
        for index, frame in tqdm(
            iter_frames(meta.path, stop=stop), total=stop, unit="frame", desc="calibration"
        ):
            if source[index] == "missing":
                video.write(overlay.draw_hud(frame.copy(), [f"frame {index}", "no homography"]))
                continue

            colour = overlay.line_colour(str(source[index]), drift[index], config.calib.drift_warn_m)
            canvas = overlay.draw_pitch_lines(frame, matrices[index], config.pitch, colour)
            drift_text = f"{drift[index]:.2f} m" if np.isfinite(drift[index]) else "n/a"
            video.write(
                overlay.draw_hud(
                    canvas, [f"frame {index}  {source[index]}", f"drift {drift_text}"]
                )
            )
    finally:
        video.release()

    print(f"Wrote {output}")
    print("Yellow pitch lines should sit on the real lines. Red marks a frame whose")
    print("fit residual exceeded the warning threshold; grey means no fit for that frame.")


def _render_tracking(paths, meta, config, stop: int) -> None:
    tracks = artifacts.read_table(paths.tracks)
    by_frame = {frame: group for frame, group in tracks.groupby("frame")}
    positions = (
        artifacts.read_table(paths.positions) if paths.positions.exists() else None
    )
    positions_by_frame = (
        {frame: group for frame, group in positions.groupby("frame")}
        if positions is not None
        else {}
    )

    output = paths.overlays_dir / "tracking_check.mp4"
    video = overlay.writer(output, meta.fps, meta.width, meta.height)

    try:
        for index, frame in tqdm(
            iter_frames(meta.path, stop=stop), total=stop, unit="frame", desc="tracking"
        ):
            canvas = frame.copy()
            group = by_frame.get(index)
            if group is not None:
                canvas = overlay.draw_tracks(
                    canvas,
                    group[["x1", "y1", "x2", "y2"]].to_numpy(),
                    group["track_id"].to_numpy(),
                )

            projected = positions_by_frame.get(index)
            if projected is not None:
                valid = projected[projected["valid"]]
                panel = overlay.minimap(
                    config.pitch,
                    valid[["x_m", "y_m"]].to_numpy(),
                    valid["track_id"].to_numpy(),
                )
                canvas = overlay.inset(canvas, panel)

            count = 0 if group is None else len(group)
            video.write(overlay.draw_hud(canvas, [f"frame {index}  tracks {count}"]))
    finally:
        video.release()

    print(f"Wrote {output}")
    if positions is None:
        print("Run 'project' first to include the pitch minimap")
