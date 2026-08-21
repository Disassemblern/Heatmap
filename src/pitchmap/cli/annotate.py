"""Click pitch landmarks on keyframes."""

from __future__ import annotations

import argparse

from pitchmap.calib import click_tool
from pitchmap.calib.keyframes import parse_keyframes, uniform_keyframes
from pitchmap.config import load_config
from pitchmap.io import artifacts
from pitchmap.io.video import read_frame


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--every-sec", type=float, default=None, help="Keyframe spacing in seconds")
    parser.add_argument("--keyframes", default=None, help="Explicit frame list, comma separated")
    parser.add_argument(
        "--add",
        default=None,
        help="Add keyframes to the existing set, comma separated (e.g. drift hotspots)",
    )


def _target_frames(args: argparse.Namespace, config, meta, existing: dict) -> tuple[int, ...]:
    if args.add:
        return tuple(sorted(set(existing) | set(parse_keyframes(args.add))))
    if args.keyframes:
        return parse_keyframes(args.keyframes)
    every_s = args.every_sec if args.every_sec is not None else config.calib.keyframe_every_s
    return tuple(sorted(set(uniform_keyframes(meta, every_s)) | set(existing)))


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)
    keyframes = artifacts.read_keyframes(paths)
    targets = _target_frames(args, config, meta, keyframes)

    print(f"Annotating {len(targets)} keyframes; {len(keyframes)} already have landmarks.")
    print("Click each named landmark, or press space if it is not visible in this frame.")

    index = 0
    action = "next"
    while 0 <= index < len(targets):
        frame_index = targets[index]
        frame = read_frame(meta.path, frame_index)
        clicked, action = click_tool.annotate_keyframe(
            frame,
            frame_index,
            config.pitch,
            keyframes.get(frame_index, {}),
            (index, len(targets)),
        )

        if clicked:
            keyframes = {**keyframes, frame_index: clicked}
        else:
            keyframes = {key: value for key, value in keyframes.items() if key != frame_index}

        artifacts.write_keyframes(paths, keyframes)
        print(f"frame {frame_index}: {len(clicked)} landmarks")
        click_tool.report_fit(clicked, config.pitch, config.calib.ransac_reproj_px)

        if action == "quit":
            break
        index += -1 if action == "prev" else 1

    click_tool.close_windows()
    annotated = {frame: clicks for frame, clicks in keyframes.items() if len(clicks) >= 4}
    artifacts.record_stage(
        paths, "annotate", {"keyframes": len(keyframes), "fittable": len(annotated)}
    )
    print(f"\nSaved {len(keyframes)} keyframes ({len(annotated)} with enough landmarks to fit)")
    print(f"Artifact: {paths.keyframes}")
