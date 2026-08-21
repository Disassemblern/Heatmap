"""Play the clip in a window with detections or tracks drawn on top.

Replays stored artifacts, so it needs no model and runs far faster than the
processing stages. Use it to inspect a finished run; use the --show flag on
detect or track to watch while those stages are working.
"""

from __future__ import annotations

import argparse

import cv2

from pitchmap.config import load_config
from pitchmap.io import artifacts
from pitchmap.io.video import iter_frames
from pitchmap.viz import overlay

WINDOW = "pitchmap preview"
KEY_QUIT = ord("q")
KEY_PAUSE = 32
KEY_BACK = ord("j")
KEY_FORWARD = ord("k")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument(
        "--kind",
        choices=("track", "detect", "calib"),
        default="track",
        help="What to draw: tracks with ids, raw detections, or the pitch model",
    )
    parser.add_argument("--start", type=int, default=0, help="First frame to show")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N frames")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (0 waits for a key press each frame)",
    )
    parser.add_argument(
        "--no-minimap", action="store_true", help="Hide the projected pitch inset"
    )


def _delay_ms(fps: float, speed: float) -> int:
    if speed <= 0:
        return 0
    return max(1, int(1000.0 / (fps * speed)))


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)

    boxes = _load_boxes(paths, args.kind)
    positions = _load_positions(paths, args.no_minimap)
    calibration = (
        artifacts.read_homographies(paths) if args.kind == "calib" else None
    )

    stop = min(meta.n_frames, args.start + args.limit) if args.limit else meta.n_frames
    delay = _delay_ms(meta.fps, args.speed)
    paused = False

    print(f"Playing {args.kind}. Space pauses, j and k step while paused, q quits.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1280, meta.width), min(720, meta.height))

    try:
        for index, frame in iter_frames(meta.path, start=args.start, stop=stop):
            canvas = _draw(frame, index, args, config, boxes, positions, calibration)
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(0 if paused else delay) & 0xFF
            while paused and key not in (KEY_PAUSE, KEY_QUIT, KEY_FORWARD, KEY_BACK):
                key = cv2.waitKey(0) & 0xFF
            if key == KEY_QUIT:
                break
            if key == KEY_PAUSE:
                paused = not paused
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyWindow(WINDOW)


def _load_boxes(paths, kind: str):
    if kind == "detect":
        table = artifacts.read_table(paths.detections)
        return {frame: group for frame, group in table.groupby("frame")}
    if kind == "track":
        table = artifacts.read_table(paths.tracks)
        return {frame: group for frame, group in table.groupby("frame")}
    return {}


def _load_positions(paths, disabled: bool):
    if disabled or not paths.positions.exists():
        return {}
    table = artifacts.read_table(paths.positions)
    return {frame: group for frame, group in table.groupby("frame")}


def _draw(frame, index, args, config, boxes, positions, calibration):
    canvas = frame.copy()
    group = boxes.get(index)

    if args.kind == "calib" and calibration is not None:
        if calibration["source"][index] != "missing":
            colour = overlay.line_colour(
                str(calibration["source"][index]),
                calibration["drift_m"][index],
                config.calib.drift_warn_m,
            )
            canvas = overlay.draw_pitch_lines(
                canvas, calibration["matrices"][index], config.pitch, colour
            )
    elif group is not None:
        xyxy = group[["x1", "y1", "x2", "y2"]].to_numpy()
        if args.kind == "track":
            canvas = overlay.draw_tracks(canvas, xyxy, group["track_id"].to_numpy())
        else:
            for x1, y1, x2, y2 in xyxy.astype(int):
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    projected = positions.get(index)
    if projected is not None:
        valid = projected[projected["valid"]]
        panel = overlay.minimap(
            config.pitch, valid[["x_m", "y_m"]].to_numpy(), valid["track_id"].to_numpy()
        )
        canvas = overlay.inset(canvas, panel)

    count = 0 if group is None else len(group)
    label = "detections" if args.kind == "detect" else args.kind
    seconds = index / 25.0
    return overlay.draw_hud(canvas, [f"frame {index}  ({seconds:.1f}s)  {label} {count}"])
