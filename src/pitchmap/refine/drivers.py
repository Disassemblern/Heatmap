"""Split and merge driver loops around the vendored GTA-Link primitives.

The loops are reimplemented here rather than calling the vendored
split_tracklets / merge_tracklets because (a) the vendored versions construct
new Tracklet objects internally and would lose the raw-row-index bookkeeping
the final output is emitted from, and (b) the team constraint has to be applied
at every distance computation - the vendored merge loop recomputes distances
after each merge, so filtering the initial matrix is not enough. The clustering
and distance primitives themselves stay vendored.

Deliberate deviation from upstream: merged tracklets are kept sorted by frame
(upstream concatenates unsorted, which feeds unordered times into its own
segment logic).
"""

from __future__ import annotations

import numpy as np

from pitchmap.config import RefineConfig
from pitchmap.refine.convert import TrackletSet
from pitchmap.refine.geometry import required_speed
from pitchmap.vendor.gta_link import Tracklet
from pitchmap.vendor.gta_link.refine_tracklets import (
    check_spatial_constraints,
    detect_id_switch,
    get_distance,
    get_spatial_constraints,
)

# The vendored detect_id_switch silently subsamples inputs above this size in a
# way that desynchronises its labels from the caller's rows; stay below it.
MAX_SWITCH_SAMPLES = 15000

TEAM_UNKNOWN = "unknown"
VETO_DISTANCE = 1.0

TeamLabels = dict[int, tuple[str, float]]


def split_pass(tracklet_set: TrackletSet, config: RefineConfig, fps: float) -> TrackletSet:
    """Split tracklets whose embeddings cluster into several identities.

    Mirrors the vendored split_tracklets loop: short tracklets pass through,
    long ones are tested with DBSCAN over their per-detection embeddings, and a
    detected switch partitions the tracklet's rows by cluster label - after the
    labels are made contiguous in time, so a split is a change point rather than
    frame-by-frame alternation.
    """
    min_run = int(round(config.min_split_s * fps))
    next_id = max(tracklet_set.tracklets) + 1
    tracklets: dict[int, Tracklet] = {}
    rows: dict[int, list[int]] = {}

    for track_id in sorted(tracklet_set.tracklets):
        tracklet = tracklet_set.tracklets[track_id]
        track_rows = tracklet_set.rows[track_id]

        if len(tracklet.times) < config.min_len_frames:
            tracklets[track_id] = tracklet
            rows[track_id] = track_rows
            continue

        labels = _switch_labels(tracklet.features, config)
        if labels is not None and min_run > 1:
            labels = _contiguous(labels, min_run)
            if len(np.unique(labels)) < 2:
                labels = None
        if labels is None:
            tracklets[track_id] = tracklet
            rows[track_id] = track_rows
            continue

        for label in np.unique(labels):
            member = np.where(labels == label)[0]
            tracklets[next_id] = _subset(tracklet, next_id, member)
            rows[next_id] = [track_rows[i] for i in member]
            next_id += 1

    return TrackletSet(tracklets=tracklets, rows=rows)


def _switch_labels(features: list[np.ndarray], config: RefineConfig) -> np.ndarray | None:
    """Cluster labels per row when an identity switch is detected, else None."""
    if len(features) <= MAX_SWITCH_SAMPLES:
        switched, labels = detect_id_switch(
            features, eps=config.eps, min_samples=config.min_samples, max_clusters=config.max_k
        )
        return labels if switched else None

    stride = int(np.ceil(len(features) / MAX_SWITCH_SAMPLES))
    sampled = list(range(0, len(features), stride))
    switched, sampled_labels = detect_id_switch(
        [features[i] for i in sampled],
        eps=config.eps,
        min_samples=config.min_samples,
        max_clusters=config.max_k,
    )
    if not switched:
        return None

    # Each unsampled row takes the label of the nearest sampled row in time.
    nearest = np.clip(
        np.round(np.arange(len(features)) / stride).astype(int), 0, len(sampled) - 1
    )
    return np.asarray(sampled_labels)[nearest]


def _contiguous(labels: np.ndarray, min_run: int) -> np.ndarray:
    """Turn per-row cluster labels into runs of at least min_run rows.

    A real identity switch is sustained: the tracker moved to another person
    and stayed there. Pose and motion blur instead make labels alternate within
    a fraction of a second. A sliding majority vote removes the alternation,
    then any run still shorter than min_run is absorbed into its longer
    neighbour. Rows stand in for frames here; a raw tracklet covers nearly
    every frame of its span.
    """
    labels = np.asarray(labels).copy()
    half = min_run // 2
    voted = labels.copy()
    for i in range(len(labels)):
        values, counts = np.unique(labels[max(0, i - half) : i + half + 1], return_counts=True)
        voted[i] = values[counts.argmax()]

    while True:
        edges = np.flatnonzero(np.diff(voted)) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, len(voted)]
        lengths = ends - starts
        if len(lengths) == 1 or lengths.min() >= min_run:
            return voted
        k = int(lengths.argmin())
        left = lengths[k - 1] if k > 0 else -1
        right = lengths[k + 1] if k + 1 < len(lengths) else -1
        neighbour = starts[k - 1] if left >= right else starts[k + 1]
        voted[starts[k] : ends[k]] = voted[neighbour]


def _subset(tracklet: Tracklet, new_id: int, member: np.ndarray) -> Tracklet:
    piece = Tracklet(
        track_id=new_id,
        frames=[tracklet.times[i] for i in member],
        scores=[tracklet.scores[i] for i in member],
        bboxes=[tracklet.bboxes[i] for i in member],
    )
    for index in member:
        piece.append_feat(tracklet.features[index])
    return piece


def merge_pass(
    tracklet_set: TrackletSet,
    config: RefineConfig,
    labels: TeamLabels | None = None,
    positions: np.ndarray | None = None,
    fps: float = 25.0,
) -> tuple[TrackletSet, dict[str, int]]:
    """Greedily reconnect tracklet fragments, mirroring the vendored merge loop.

    Repeatedly merge the globally closest pair below merge_dist_thres, subject
    to the vendored spatial check, the team veto, and - when calibration is
    available - a physical speed limit in pitch metres. Returns the merged set
    and a count of what each constraint rejected.

    The speed limit matters more than the other two. The vendored spatial check
    compares image-space box ranges, which a panning camera makes almost
    vacuous: without the pitch-space gate, merges on this footage implied
    players moving at up to 140 m/s.
    """
    tracklets = dict(tracklet_set.tracklets)
    rows = {tid: list(indices) for tid, indices in tracklet_set.rows.items()}
    labels = dict(labels) if labels else None
    rejected = {"team": 0, "speed": 0, "spatial": 0}
    speed_gate = positions if config.use_speed_constraint else None

    max_x_range, max_y_range = get_spatial_constraints(tracklets, config.spatial_factor)

    ids = sorted(tracklets)
    distance = {}
    for position, first in enumerate(ids):
        for second in ids[position + 1 :]:
            value, reason = _pair_distance(
                tracklets, rows, first, second, labels, config, speed_gate, fps
            )
            distance[(first, second)] = value
            if reason:
                rejected[reason] += 1

    while True:
        candidates = [(value, pair) for pair, value in distance.items() if value < config.merge_dist_thres]
        if not candidates:
            break
        _, (first, second) = min(candidates)

        if not check_spatial_constraints(
            tracklets[first], tracklets[second], max_x_range, max_y_range
        ):
            distance[(first, second)] = config.merge_dist_thres
            rejected["spatial"] += 1
            continue

        _merge_into(tracklets, rows, labels, first, second)
        for pair in [p for p in distance if second in p]:
            del distance[pair]
        for other in tracklets:
            if other == first:
                continue
            pair = (min(first, other), max(first, other))
            value, reason = _pair_distance(
                tracklets, rows, first, other, labels, config, speed_gate, fps
            )
            distance[pair] = value
            if reason:
                rejected[reason] += 1

    return TrackletSet(tracklets=tracklets, rows=rows), rejected


def _pair_distance(
    tracklets: dict[int, Tracklet],
    rows: dict[int, list[int]],
    first_id: int,
    second_id: int,
    labels: TeamLabels | None,
    config: RefineConfig,
    positions: np.ndarray | None,
    fps: float,
) -> tuple[float, str | None]:
    """Vendored appearance distance, with the hard constraints applied on top.

    Returns the distance and which constraint rejected the pair, if any.
    """
    first, second = tracklets[first_id], tracklets[second_id]

    if labels is not None and _vetoed(first_id, second_id, labels, config.min_veto_confidence):
        return VETO_DISTANCE, "team"

    if positions is not None:
        speed = required_speed(
            first.times, rows[first_id], second.times, rows[second_id], positions, fps
        )
        if speed > config.max_speed_ms:
            return VETO_DISTANCE, "speed"

    return float(get_distance(first_id, second_id, first, second)), None


def _vetoed(
    first_id: int,
    second_id: int,
    labels: TeamLabels,
    min_confidence: float,
) -> bool:
    """Two tracklets are different people when both confidently differ in team."""
    team_a, confidence_a = labels.get(first_id, (TEAM_UNKNOWN, 0.0))
    team_b, confidence_b = labels.get(second_id, (TEAM_UNKNOWN, 0.0))

    if TEAM_UNKNOWN in (team_a, team_b) or team_a == team_b:
        return False
    return confidence_a >= min_confidence and confidence_b >= min_confidence


def _merge_into(
    tracklets: dict[int, Tracklet],
    rows: dict[int, list[int]],
    labels: TeamLabels | None,
    first: int,
    second: int,
) -> None:
    """Absorb tracklet `second` into `first`, keeping frame order."""
    target, source = tracklets[first], tracklets[second]
    order = np.argsort(target.times + source.times, kind="stable")

    times = target.times + source.times
    scores = target.scores + source.scores
    bboxes = target.bboxes + source.bboxes
    features = target.features + source.features
    merged_rows = rows[first] + rows[second]

    merged = Tracklet(
        track_id=first,
        frames=[times[i] for i in order],
        scores=[scores[i] for i in order],
        bboxes=[bboxes[i] for i in order],
    )
    for index in order:
        merged.append_feat(features[index])

    tracklets[first] = merged
    rows[first] = [merged_rows[i] for i in order]
    del tracklets[second], rows[second]

    if labels is not None:
        kept = max(
            labels.get(first, (TEAM_UNKNOWN, 0.0)),
            labels.get(second, (TEAM_UNKNOWN, 0.0)),
            key=lambda item: item[1],
        )
        labels[first] = kept
        labels.pop(second, None)
