"""Cluster tracklets into two teams from jersey colours.

Two different questions need two different features.

Which of the two teams a tracklet belongs to is decided in chroma (the a and b
channels of Lab) when that separates the kits better than full Lab, which it
usually does: lightness mostly tracks shadows and floodlighting rather than kit
identity.

Whether a tracklet is a player at all is decided in full Lab, because lightness
is exactly what sets officials apart. Measured on the test clip: the navy team
sits at L 81 and the red team at L 124, while referees, keepers and staff sit at
L 189. A cyan referee is nearly indistinguishable from navy in chroma alone, and
obvious once lightness is included.

Distances are expressed as a fraction of the gap between the two team centroids,
so the threshold does not depend on how contrasting the two kits happen to be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pitchmap.config import TeamConfig
from pitchmap.team.jersey import JerseyFeatures

TEAM_A = "A"
TEAM_B = "B"
TEAM_OTHER = "other"
TEAM_UNKNOWN = "unknown"

CHROMA = "chroma"
FULL_LAB = "lab"
FEATURE_SPACES = {CHROMA: slice(1, 3), FULL_LAB: slice(0, 3)}
MIN_FIT_TRACKLETS = 4


def project(colours: np.ndarray, space: str) -> np.ndarray:
    return colours[:, FEATURE_SPACES[space]]


def choose_space(colours: np.ndarray, fit_mask: np.ndarray) -> tuple[str, float]:
    """Pick the feature space whose two clusters are best separated.

    Returns the space name and its silhouette score.
    """
    scored = []
    for space in (CHROMA, FULL_LAB):
        points = project(colours, space)[fit_mask]
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(points)
        if len(np.unique(labels)) < 2:
            continue
        scored.append((float(silhouette_score(points, labels)), space))

    if not scored:
        return CHROMA, 0.0
    score, space = max(scored)
    return space, score


def tracklet_features(
    tracks: pd.DataFrame, features: JerseyFeatures
) -> tuple[pd.DataFrame, np.ndarray]:
    """Pixel-count-weighted mean Lab colour per tracklet."""
    table = pd.DataFrame(
        {
            "track_id": tracks["track_id"].iloc[features.row_index].to_numpy(),
            "n_pixels": features.n_pixels,
        }
    )
    usable = table["n_pixels"] > 0

    rows = []
    colours = []
    for track_id, group in table[usable].groupby("track_id"):
        member = group.index.to_numpy()
        weights = features.n_pixels[member].astype(np.float64)
        colours.append(np.average(features.lab[member], axis=0, weights=weights))
        rows.append({"track_id": int(track_id), "n_rows": len(member)})

    return pd.DataFrame(rows), np.array(colours, dtype=np.float64)


def fit_centroids(
    colours: np.ndarray,
    fit_mask: np.ndarray,
    space: str,
    config: TeamConfig,
) -> np.ndarray:
    """Fit the two team centroids, in full Lab, then refit without the outliers.

    Clustering happens in the chosen space, but the centroids are returned as
    full Lab colours so callers can ask both questions: which team (projected)
    and whether it is a team at all (full Lab).

    Officials and goalkeepers drag a centroid towards themselves on the first
    pass; excluding whatever the first pass considers "other" and refitting
    gives centroids that describe the two teams alone.
    """
    points = project(colours, space)
    fitted, fitted_lab = points[fit_mask], colours[fit_mask]

    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(fitted)
    centroids = _lab_centroids(fitted_lab, labels)

    ratios = _distance_ratios(fitted_lab, centroids)[0]
    inliers = ratios <= config.other_dist_ratio
    if inliers.sum() < MIN_FIT_TRACKLETS or inliers.all():
        return centroids

    refit_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(fitted[inliers])
    # Keep the refit only if it still describes two separated groups. Dropping
    # outliers can leave a subset that splits along a meaningless axis, which
    # would push most tracklets onto one centroid.
    if len(np.unique(refit_labels)) < 2:
        return centroids
    if silhouette_score(fitted[inliers], refit_labels) < config.min_silhouette:
        return centroids
    return _lab_centroids(fitted_lab[inliers], refit_labels)


def _lab_centroids(colours_lab: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean full-Lab colour of each cluster."""
    return np.stack([colours_lab[labels == label].mean(axis=0) for label in (0, 1)])


def _distance_ratios(
    points: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance to each centroid, and to the nearer one as a fraction of their gap."""
    separation = float(np.linalg.norm(centroids[0] - centroids[1]))
    dist_a = np.linalg.norm(points - centroids[0], axis=1)
    dist_b = np.linalg.norm(points - centroids[1], axis=1)
    ratio = np.minimum(dist_a, dist_b) / max(separation, 1e-9)
    return ratio, dist_a, dist_b


def label_colours(
    per_tracklet: pd.DataFrame,
    colours: np.ndarray,
    centroids: np.ndarray,
    space: str,
    config: TeamConfig,
) -> pd.DataFrame:
    """Label tracklets against fitted centroids."""
    # Which team: in the space that separates the two kits best.
    projected = project(colours, space)
    projected_centroids = project(centroids, space)
    _, near_a, near_b = _distance_ratios(projected, projected_centroids)
    nearest = np.where(near_a <= near_b, TEAM_A, TEAM_B)

    # Player or not: in full Lab, where officials stand apart by lightness.
    ratio, dist_a, dist_b = _distance_ratios(colours, centroids)
    separation = float(np.linalg.norm(centroids[0] - centroids[1]))

    team = np.where(ratio > config.other_dist_ratio, TEAM_OTHER, nearest)
    team = np.where(per_tracklet["n_rows"].to_numpy() < config.min_rows, TEAM_UNKNOWN, team)

    return per_tracklet.assign(
        team=team,
        dist_a=dist_a,
        dist_b=dist_b,
        confidence=np.clip(np.abs(dist_a - dist_b) / max(separation, 1e-9), 0.0, 1.0),
    )


def assign_teams(
    per_tracklet: pd.DataFrame,
    colours: np.ndarray,
    fit_ids: set[int],
    config: TeamConfig,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Cluster tracklets into two teams. Returns (labels, centroids, space).

    When neither feature space separates the kits, every tracklet is labelled
    unknown so the downstream merge constraint degrades to off rather than
    acting on labels that are close to coin flips.
    """
    fit_mask = per_tracklet["track_id"].isin(fit_ids).to_numpy()
    if fit_mask.sum() < MIN_FIT_TRACKLETS:
        fit_mask = np.ones(len(per_tracklet), dtype=bool)

    space, score = choose_space(colours, fit_mask)
    print(f"Clustering in {space} (silhouette {score:.2f})")

    if score < config.min_silhouette:
        print(
            f"Warning: kits do not separate (silhouette {score:.2f} below "
            f"{config.min_silhouette}); labelling everything unknown."
        )
        labels = per_tracklet.assign(
            team=TEAM_UNKNOWN, dist_a=np.nan, dist_b=np.nan, confidence=0.0
        )
        return labels, np.zeros((2, 3)), space

    centroids = fit_centroids(colours, fit_mask, space, config)
    return label_colours(per_tracklet, colours, centroids, space, config), centroids, space


def label_rows(
    rows_per_tracklet: dict[int, list[int]],
    features: JerseyFeatures,
    centroids: np.ndarray,
    space: str,
    config: TeamConfig,
) -> pd.DataFrame:
    """Label tracklets defined by raw row indices against existing centroids.

    Used after refinement, when splitting and merging have changed which rows
    belong together: labels are re-derived for the final grouping rather than
    inherited from tracklets whose membership no longer holds.
    """
    by_row = {int(row): index for index, row in enumerate(features.row_index)}

    ids: list[int] = []
    counts: list[int] = []
    colours: list[np.ndarray] = []
    empty: list[int] = []

    for track_id, rows in sorted(rows_per_tracklet.items()):
        member = [by_row[row] for row in rows if row in by_row]
        member = [i for i in member if features.n_pixels[i] > 0]
        if not member:
            empty.append(track_id)
            continue
        weights = features.n_pixels[member].astype(np.float64)
        colours.append(np.average(features.lab[member], axis=0, weights=weights))
        ids.append(track_id)
        counts.append(len(member))

    labelled = pd.DataFrame()
    if ids:
        labelled = label_colours(
            pd.DataFrame({"track_id": ids, "n_rows": counts}),
            np.array(colours, dtype=np.float64),
            centroids,
            space,
            config,
        )

    if empty:
        blank = pd.DataFrame(
            {
                "track_id": empty,
                "n_rows": 0,
                "team": TEAM_UNKNOWN,
                "dist_a": np.nan,
                "dist_b": np.nan,
                "confidence": 0.0,
            }
        )
        labelled = pd.concat([labelled, blank], ignore_index=True)

    return labelled.sort_values("track_id", ignore_index=True)
