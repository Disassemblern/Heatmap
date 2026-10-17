"""Cut a long recording into overlapping clips, each processed as its own run.

A half is easier to check in 5-minute pieces than in one overnight run, and
every piece is an ordinary clip afterwards: init-run and the stages treat it
like any other video.

Consecutive clips overlap by a few seconds so the `link` stage can carry player
names from one clip into the next: the shared frames show the same players in
the same places, and matching tracklets there needs no guessing.

ffmpeg copies the video stream instead of re-encoding it, which is instant and
leaves the pixels untouched, so the detector finds identical boxes in the
overlap of both clips. The catch is that a copied clip can only start on a
keyframe (every 10 s on the test clip). Each clip therefore starts on the
keyframe at or before the requested time, and its exact start is written to a
sidecar <clip>.json that init-run picks up, so frames line up exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from pitchmap.io.video import read_meta


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, help="Recording to cut")
    parser.add_argument(
        "--every", type=float, default=None, help="Cut the whole video into parts of this many minutes"
    )
    parser.add_argument(
        "--overlap", type=float, default=5.0, help="Seconds each part shares with the next (default 5)"
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


def plan_clips(
    keyframes: list[float], duration: float, every: float, overlap: float
) -> list[tuple[float, float]]:
    """(start, end) of each clip: starts on keyframes, each overlapping the next.

    A requested start between keyframes moves back to the one before it. When
    parts are shorter than the keyframe spacing, several requests land on the
    same keyframe and collapse into one clip. A final part that would add no
    frames beyond the previous clip's overlap is dropped.
    """
    starts: list[float] = []
    requested = 0.0
    while requested < duration:
        start = max(k for k in keyframes if k <= requested + 1e-6)
        if duration - start <= overlap:
            break
        if not starts or start > starts[-1]:
            starts.append(start)
        requested += every
    ends = [start + overlap for start in starts[1:]] + [duration]
    return list(zip(starts, ends))


def keyframe_times(video: Path) -> list[float]:
    """Presentation times of the video's keyframes, where a copied clip can start."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
            "-of", "csv=p=0", str(video),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed:\n{result.stderr.strip()}")
    times = []
    for line in result.stdout.splitlines():
        token = line.split(",")[0].strip()
        try:
            times.append(float(token))
        except ValueError:
            continue
    if not times:
        raise SystemExit(f"No keyframes found in {video}.")
    return sorted(times)


def _label(seconds: float) -> str:
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}m{rest:02d}s"


def _require_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is not on PATH. Install ffmpeg (https://ffmpeg.org) and try again.")


def _cut(video: Path, start: float, end: float, target: Path, fps: float) -> None:
    # Seek half a frame past the keyframe: ffmpeg then starts on that keyframe,
    # where seeking to its exact time could round back to the previous one.
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start + 0.5 / fps:.6f}", "-i", str(video),
            "-t", f"{end - start:.6f}",
            "-map", "0", "-c", "copy", "-avoid_negative_ts", "make_zero",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {target.name}:\n{result.stderr.strip()}")


def run(args: argparse.Namespace) -> None:
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"No such video: {video}")
    if (args.every is None) == (args.start is None):
        raise SystemExit("Give either --every MINUTES, or --start (and optionally --end).")
    if args.overlap < 0:
        raise SystemExit("--overlap cannot be negative.")
    _require_tools()

    meta = read_meta(video)
    keyframes = keyframe_times(video)
    out_dir = Path(args.out_dir) if args.out_dir else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.every is not None:
        if args.every * 60 <= args.overlap:
            raise SystemExit("--every must be longer than --overlap.")
        plan = plan_clips(keyframes, meta.duration_s, args.every * 60, args.overlap)
        targets = [out_dir / f"{video.stem}_p{i:02d}.mp4" for i in range(1, len(plan) + 1)]
    else:
        requested = parse_time(args.start)
        end = parse_time(args.end) if args.end else meta.duration_s
        if end <= requested:
            raise SystemExit(f"--end ({end:.0f} s) must come after --start ({requested:.0f} s).")
        start = max(k for k in keyframes if k <= requested + 1e-6)
        plan = [(start, min(end, meta.duration_s))]
        targets = [out_dir / f"{video.stem}_{_label(start)}-{_label(plan[0][1])}.mp4"]

    # A clip's run directory is named after it, so replacing the clip under an
    # existing run would leave that run's artifacts describing different frames.
    existing = [target for target in targets if target.exists()]
    if existing and not args.force:
        names = ", ".join(path.name for path in existing[:5])
        raise SystemExit(f"Clips already exist ({names}). Pass --force to overwrite them.")

    if len(keyframes) > 1:
        print(f"Keyframes every {keyframes[1] - keyframes[0]:.1f} s; each clip starts on one.")
    for (start, end), target in zip(plan, targets):
        _cut(video, start, end, target, meta.fps)
        sidecar = {"source": video.name, "start_s": start, "end_s": end, "overlap_s": args.overlap}
        target.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        frames = read_meta(target).n_frames
        print(f"  {target.name:34s} {_label(start)} - {_label(end)}  ({frames} frames)")
    print(f"Wrote {len(plan)} clip(s) to {out_dir}, each with a .json holding its exact start.")
    print("Process each one like any clip, starting with: python -m pitchmap init-run --video <clip>")
