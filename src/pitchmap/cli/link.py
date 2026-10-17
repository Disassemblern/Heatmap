"""Carry player names from one clip into the next through their overlap.

Consecutive clips from `cut` share a few seconds of video, frame for frame. The
stream was copied rather than re-encoded, so those frames are bit-identical and
the detector found the same boxes in both clips. A tracklet in the next clip is
the continuation of whichever tracklet in the earlier clip its boxes coincide
with across the overlap, and it inherits that tracklet's name.

Names are written into the next clip's review.csv only where its player cell is
empty. A name you typed yourself is never overwritten; a disagreement is
reported instead, since one of the two is wrong and only you can say which.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pitchmap.config import load_config
from pitchmap.eval import review
from pitchmap.io import artifacts

BOX = ["x1", "y1", "x2", "y2"]
# Two boxes are the same player in the same frame when they overlap this much.
MIN_IOU = 0.5
# A tracklet continues another only if their boxes coincide in at least this
# share of the overlap frames the later tracklet appears in.
MIN_SHARE = 0.5
# Frames two pieces of one player may share at a handover.
MAX_SHARED_FRAMES = 2
# Identical frames give identical detections; anything well below this means
# the two clips are not showing the same frames at the recorded offset.
MIN_ALIGNED = 0.9


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prev", required=True, help="Run of the earlier clip, named in its review.csv")
    parser.add_argument("--run", required=True, help="Run of the next clip, to receive the names")


def frame_offset(prev_meta: dict, run_meta: dict) -> int:
    """Index in the earlier clip of the next clip's first frame."""
    prev_clip, run_clip = prev_meta.get("clip"), run_meta.get("clip")
    if prev_clip is None or run_clip is None:
        raise SystemExit(
            "Both runs must come from clips made by 'cut': their meta.json records no "
            "clip start. Re-run init-run on the clips, with their .json files beside them."
        )
    if prev_clip["source"] != run_clip["source"]:
        raise SystemExit(
            f"The clips come from different recordings ({prev_clip['source']} and {run_clip['source']})."
        )
    offset = round((run_clip["start_s"] - prev_clip["start_s"]) * prev_meta["video"]["fps"])
    if not 0 < offset < prev_meta["video"]["n_frames"]:
        raise SystemExit("These clips do not overlap. Pass the earlier clip as --prev and the next as --run.")
    return offset


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (n, 4) and (m, 4) xyxy boxes."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _pairs(a: np.ndarray, b: np.ndarray, min_iou: float) -> list[tuple[int, int]]:
    """Index pairs of boxes that coincide, one to one."""
    if not len(a) or not len(b):
        return []
    iou = _iou(a, b)
    rows, cols = linear_sum_assignment(-iou)
    return [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]


def _overlap_frames(table: pd.DataFrame, n_overlap: int) -> dict[int, pd.DataFrame]:
    return {int(f): g for f, g in table[table["frame"] < n_overlap].groupby("frame")}


def aligned_share(prev_det: pd.DataFrame, run_det: pd.DataFrame, offset: int, n_overlap: int) -> float:
    """Share of the next clip's overlap detections found at the same place in the earlier clip."""
    prev_by = {int(f): g for f, g in prev_det[prev_det["frame"] >= offset].groupby("frame")}
    found = total = 0
    for frame, group in _overlap_frames(run_det, n_overlap).items():
        total += len(group)
        prev = prev_by.get(frame + offset)
        if prev is not None:
            found += len(_pairs(group[BOX].to_numpy(), prev[BOX].to_numpy(), 0.9))
    return found / total if total else 0.0


def link_tracklets(
    prev_tracks: pd.DataFrame, run_tracks: pd.DataFrame, offset: int, n_overlap: int
) -> dict[int, int]:
    """Next clip's track id -> the earlier clip's track id it continues."""
    prev_by = {int(f): g for f, g in prev_tracks[prev_tracks["frame"] >= offset].groupby("frame")}
    votes: Counter = Counter()
    frames_of: dict[int, set[int]] = defaultdict(set)
    for frame, group in _overlap_frames(run_tracks, n_overlap).items():
        for track_id in group["track_id"]:
            frames_of[int(track_id)].add(frame)
        prev = prev_by.get(frame + offset)
        if prev is None:
            continue
        for r, c in _pairs(group[BOX].to_numpy(), prev[BOX].to_numpy(), MIN_IOU):
            votes[(int(group["track_id"].iloc[r]), int(prev["track_id"].iloc[c]))] += 1

    best: dict[int, tuple[int, int]] = {}
    for (run_id, prev_id), count in votes.items():
        if count >= MIN_SHARE * len(frames_of[run_id]) and count > best.get(run_id, (0, 0))[1]:
            best[run_id] = (prev_id, count)

    # Several tracklets can continue one earlier tracklet when they take turns:
    # refine sometimes cuts one player into consecutive pieces. Two tracklets on
    # screen at the same time are two people, so only the better-supported one
    # inherits; a handover frame or two where both boxes exist is tolerated.
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for run_id, (prev_id, count) in best.items():
        candidates[prev_id].append((count, run_id))
    links: dict[int, int] = {}
    for prev_id, ranked in candidates.items():
        taken: set[int] = set()
        for _, run_id in sorted(ranked, reverse=True):
            if len(frames_of[run_id] & taken) <= MAX_SHARED_FRAMES:
                links[run_id] = prev_id
                taken |= frames_of[run_id]
    return links


def carry_names(
    judged: pd.DataFrame, links: dict[int, int], names: dict[int, str]
) -> tuple[pd.DataFrame, list, list, list]:
    """Fill empty player cells from linked names.

    Returns the updated table, plus (track_id, name) for names carried, for
    linked tracklets with no row to hold a name, and (track_id, yours, carried)
    for disagreements, where your entry is kept.
    """
    updated = judged.copy()
    rows = {int(t): i for i, t in zip(updated.index, updated["track_id"])}
    carried, unlisted, conflicts = [], [], []
    for run_id, prev_id in sorted(links.items()):
        name = names.get(prev_id)
        if not name:
            continue
        if run_id not in rows:
            unlisted.append((run_id, name))
            continue
        current = updated.at[rows[run_id], "player"]
        if current == "":
            updated.at[rows[run_id], "player"] = name
            carried.append((run_id, name))
        elif current.casefold() != name.casefold():
            conflicts.append((run_id, current, name))
    return updated, carried, unlisted, conflicts


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prev, cur = artifacts.run_paths(args.prev), artifacts.run_paths(args.run)
    prev_meta, cur_meta = artifacts.read_meta(prev), artifacts.read_meta(cur)
    offset = frame_offset(prev_meta, cur_meta)
    n_overlap = prev_meta["video"]["n_frames"] - offset
    fps = prev_meta["video"]["fps"]

    share = aligned_share(
        artifacts.read_table(prev.detections), artifacts.read_table(cur.detections), offset, n_overlap
    )
    print(f"Overlap: {n_overlap} frames ({n_overlap / fps:.1f} s); {share:.0%} of its detections match")
    if share < MIN_ALIGNED:
        raise SystemExit(
            "The clips do not show the same frames at the recorded offset, so tracklets "
            "cannot be matched. Were both clips cut from the same recording by 'cut'?"
        )

    for paths in (prev, cur):
        if not paths.review.exists():
            raise SystemExit(f"{paths.root.name} has no review.csv yet. Run 'review' on it first.")
    prev_tracks, cur_tracks = artifacts.read_table(prev.tracks), artifacts.read_table(cur.tracks)
    review.check_key(prev.review_key, prev_tracks)
    review.check_key(cur.review_key, cur_tracks)

    links = link_tracklets(prev_tracks, cur_tracks, offset, n_overlap)
    prev_judged = review.read_reviews(prev.review)
    names = {
        int(t): p
        for t, p, v in zip(prev_judged["track_id"], prev_judged["player"], prev_judged["verdict"])
        if p and v != review.VERDICT_MIXED
    }

    updated, carried, unlisted, conflicts = carry_names(review.read_reviews(cur.review), links, names)
    updated.to_csv(cur.review, index=False)
    artifacts.record_stage(
        cur,
        "link",
        {
            "prev": prev.root.name,
            "offset_frames": offset,
            "overlap_frames": n_overlap,
            "aligned_share": share,
            "linked": len(links),
            "carried": len(carried),
            "conflicts": len(conflicts),
        },
        config,
    )

    print(f"Linked {len(links)} tracklets across the overlap; {len(names)} were named in {prev.root.name}")
    for run_id, name in carried:
        print(f"  tracklet {run_id:<4} <- {name}")
    if unlisted:
        items = ", ".join(f"{t} ({n})" for t, n in unlisted)
        print(f"Not carried, too short to have a review.csv row: {items}")
    for run_id, yours, theirs in conflicts:
        print(f"Kept your name for tracklet {run_id}: '{yours}', but it continues '{theirs}'. One is wrong.")
    print(f"Artifact: {cur.review}")
