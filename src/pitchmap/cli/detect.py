"""Detect players in every frame and store them at a low confidence floor."""

from __future__ import annotations

import argparse

import cv2
import pandas as pd
from tqdm import tqdm

from pitchmap.config import load_config, with_overrides
from pitchmap.detect.yolo import YoloDetector
from pitchmap.io import artifacts
from pitchmap.io.video import iter_frames
from pitchmap.viz.overlay import draw_hud

WINDOW = "pitchmap detect"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--model", default=None, help="Override the detector weights path")
    parser.add_argument("--conf", type=float, default=None, help="Override the confidence floor")
    parser.add_argument("--imgsz", type=int, default=None, help="Override the inference size")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N frames (smoke test)")
    parser.add_argument(
        "--show", action="store_true", help="Display detections in a window while running"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing longer run"
    )


def _display(frame, detections, index: int) -> bool:
    """Draw detections in a window. Returns False when the user asks to stop."""
    canvas = frame.copy()
    for x1, y1, x2, y2 in detections.xyxy.astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    canvas = draw_hud(canvas, [f"frame {index}  detections {len(detections.xyxy)}"])
    cv2.imshow(WINDOW, canvas)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def run(args: argparse.Namespace) -> None:
    config = with_overrides(
        load_config(args.config),
        "detect",
        model_path=args.model,
        conf_store=args.conf,
        imgsz=args.imgsz,
    )
    paths = artifacts.run_paths(args.run)
    meta = artifacts.video_meta(paths)
    detector = YoloDetector(config.detect)
    stop = min(meta.n_frames, args.limit) if args.limit else meta.n_frames

    artifacts.guard_truncation(paths.detections, stop, args.force)

    print(f"Detecting with {detector.describe()}")
    if args.show:
        print("Showing detections. Press q to stop early; work so far is still saved.")

    records = []
    for index, frame in tqdm(iter_frames(meta.path, stop=stop), total=stop, unit="frame"):
        detections = detector.detect(frame)

        if args.show and not _display(frame, detections, index):
            stop = index + 1
            break

        records.extend(
            {
                "frame": index,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(conf),
                "cls": int(class_id),
            }
            for (x1, y1, x2, y2), conf, class_id in zip(
                detections.xyxy, detections.confidence, detections.class_id, strict=True
            )
        )

    if args.show:
        cv2.destroyWindow(WINDOW)

    table = pd.DataFrame.from_records(records)
    artifacts.write_table(table, paths.detections)
    artifacts.record_stage(
        paths,
        "detect",
        {"detector": detector.describe(), "frames": stop, "detections": len(table)},
    )

    per_frame = len(table) / stop if stop else 0.0
    print(f"Wrote {len(table)} detections over {stop} frames ({per_frame:.1f} per frame)")
    print(f"Artifact: {paths.detections}")
