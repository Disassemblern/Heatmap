"""Command dispatch for `python -m pitchmap <stage>`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from pitchmap.cli import (
    annotate,
    calibrate,
    detect,
    heatmap,
    init_run,
    overlay,
    preview,
    project,
    track,
)

STAGES: dict[str, tuple[Callable[[argparse.ArgumentParser], None], Callable[[argparse.Namespace], None], str]] = {
    "init-run": (init_run.add_arguments, init_run.run, "Create a run directory for a clip"),
    "detect": (detect.add_arguments, detect.run, "Detect players in every frame"),
    "track": (track.add_arguments, track.run, "Assign track ids across frames"),
    "annotate": (annotate.add_arguments, annotate.run, "Click pitch landmarks on keyframes"),
    "calibrate": (calibrate.add_arguments, calibrate.run, "Fit and propagate per-frame homographies"),
    "project": (project.add_arguments, project.run, "Project foot points into pitch metres"),
    "heatmap": (heatmap.add_arguments, heatmap.run, "Render per-tracklet heatmaps"),
    "overlay": (overlay.add_arguments, overlay.run, "Render calibration or tracking sanity videos"),
    "preview": (preview.add_arguments, preview.run, "Play the clip in a window with results drawn"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pitchmap", description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    for name, (add_arguments, handler, help_text) in STAGES.items():
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument(
            "--config",
            default=None,
            help="Path to a YAML config file (default: configs/default.yaml)",
        )
        add_arguments(subparser)
        subparser.set_defaults(handler=handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
