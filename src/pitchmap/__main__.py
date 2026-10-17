"""Command dispatch for `python -m pitchmap <stage>`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from pitchmap.cli import (
    annotate,
    calibrate,
    combine,
    cut,
    detect,
    eval as eval_stage,
    heatmap,
    init_run,
    link,
    overlay,
    preview,
    project,
    refine,
    review,
    team,
    track,
)

STAGES: dict[str, tuple[Callable[[argparse.ArgumentParser], None], Callable[[argparse.Namespace], None], str]] = {
    "cut": (cut.add_arguments, cut.run, "Cut a long recording into shorter clips"),
    "init-run": (init_run.add_arguments, init_run.run, "Create a run directory for a clip"),
    "detect": (detect.add_arguments, detect.run, "Detect players in every frame"),
    "track": (track.add_arguments, track.run, "Assign track ids across frames"),
    "team": (team.add_arguments, team.run, "Cluster tracklets into teams by jersey colour"),
    "refine": (refine.add_arguments, refine.run, "Split and reconnect tracklets offline"),
    "annotate": (annotate.add_arguments, annotate.run, "Click pitch landmarks on keyframes"),
    "calibrate": (calibrate.add_arguments, calibrate.run, "Fit a homography per frame from detected pitch keypoints"),
    "project": (project.add_arguments, project.run, "Project foot points into pitch metres"),
    "heatmap": (heatmap.add_arguments, heatmap.run, "Render per-tracklet heatmaps"),
    "review": (review.add_arguments, review.run, "Render tracklet sheets for human judgement"),
    "eval": (eval_stage.add_arguments, eval_stage.run, "Score identity against human judgements"),
    "link": (link.add_arguments, link.run, "Carry player names into the next clip through the overlap"),
    "combine": (combine.add_arguments, combine.run, "One heatmap per named player across runs"),
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
