"""Cut a long recording into shorter clips, each processed as its own run.

A half is easier to check in 5-minute pieces than in one overnight run, and
every piece is an ordinary clip afterwards: init-run and the stages treat it
like any other video, so the pipeline itself needs no notion of time ranges.

ffmpeg copies the video stream instead of re-encoding it, which is instant and
leaves the pixels the detector sees untouched. The cost is that cuts land on
the nearest keyframe, usually within a second or two of the requested time;
for heatmaps that does not matter.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from pitchmap.io.video import read_meta


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, help="Recording to cut")
    parser.add_argument(
        "--every", type=float, default=None, help="Cut the whole video into parts of this many minutes"
    )
    parser.add_argument("--start", default=None, help="Start of one range, as mm:ss, hh:mm:ss or seconds")
    parser.add_argument("--end", default=None, help="End of that range, same formats")
    parser.add_argument(
        "--out-dir", default=None, help="Where to write the clips (default: next to the video)"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite clips that already exist")


def parse_time(text: str) -> float:
    """Seconds from 'ss', 'mm:ss' or 'hh:mm:ss'."""
    seconds = 0.0
    for part in text.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def _label(seconds: float) -> str:
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}m{rest:02d}s"


def _ffmpeg(arguments: list[str]) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH. Install it (https://ffmpeg.org) and try again.")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{result.stderr.strip()}")


def _refuse_overwrite(existing: list[Path], force: bool) -> None:
    # A clip's run directory is named after it, so replacing the clip under an
    # existing run would leave that run's artifacts describing different frames.
    if existing and not force:
        names = ", ".join(path.name for path in existing[:5])
        raise SystemExit(f"Clips already exist ({names}). Pass --force to overwrite them.")


def run(args: argparse.Namespace) -> None:
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"No such video: {video}")
    if (args.every is None) == (args.start is None):
        raise SystemExit("Give either --every MINUTES, or --start (and optionally --end).")

    out_dir = Path(args.out_dir) if args.out_dir else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.every is not None:
        if args.every <= 0:
            raise SystemExit("--every must be a positive number of minutes.")
        _refuse_overwrite(sorted(out_dir.glob(f"{video.stem}_p[0-9][0-9].mp4")), args.force)
        _ffmpeg(
            [
                "-i", str(video),
                "-map", "0", "-c", "copy",
                "-f", "segment",
                "-segment_time", str(args.every * 60),
                "-segment_start_number", "1",
                "-reset_timestamps", "1",
                str(out_dir / f"{video.stem}_p%02d.mp4"),
            ]
        )
        clips = sorted(out_dir.glob(f"{video.stem}_p[0-9][0-9].mp4"))
    else:
        start = parse_time(args.start)
        end = parse_time(args.end) if args.end else read_meta(video).duration_s
        if end <= start:
            raise SystemExit(f"--end ({end:.0f} s) must come after --start ({start:.0f} s).")
        target = out_dir / f"{video.stem}_{_label(start)}-{_label(end)}.mp4"
        _refuse_overwrite([target] if target.exists() else [], args.force)
        _ffmpeg(
            [
                "-ss", str(start), "-i", str(video),
                "-t", str(end - start),
                "-map", "0", "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(target),
            ]
        )
        clips = [target]

    offset = parse_time(args.start) if args.start else 0.0
    print(f"Wrote {len(clips)} clip(s) to {out_dir}:")
    for clip in clips:
        meta = read_meta(clip)
        print(f"  {clip.name:32s} {_label(offset)} - {_label(offset + meta.duration_s)}  ({meta.n_frames} frames)")
        offset += meta.duration_s
    print("Process each one like any clip, starting with: python -m pitchmap init-run --video <clip>")
