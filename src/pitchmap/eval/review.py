"""Ground-truth judgements about tracklets, and the metrics computed from them.

Automatic purity is measured with the same jersey colours that produce the team
labels, so it is blind to a swap between two players of the same team. A human
looking at crops from across a tracklet's life is not, which is what makes these
judgements worth collecting.

The schema is deliberately small, because the annotation cost is what limits how
much ground truth exists:

    track_id  the tracklet judged
    verdict   clean | mixed | unsure
    team      A | B | other | unknown, as judged from the crops
    player    optional free text, a shirt number or a name
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

def fingerprint(tracks: pd.DataFrame) -> str:
    """Stable descriptor of a set of tracklets.

    Judgements are keyed to track ids, and refine renumbers those every time it
    runs, so ground truth silently stops describing the tracklets it was made
    against. Storing a fingerprint turns that into a loud error instead of a
    quietly wrong score.
    """
    summary = (
        tracks.groupby("track_id")
        .agg(rows=("frame", "size"), first=("frame", "min"))
        .sort_index()
    )
    payload = summary.to_csv().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def write_key(path: Path, tracks: pd.DataFrame) -> None:
    path.write_text(
        json.dumps({"fingerprint": fingerprint(tracks), "tracklets": int(tracks["track_id"].nunique())}),
        encoding="utf-8",
    )


def check_key(path: Path, tracks: pd.DataFrame) -> None:
    """Refuse to score judgements made against a different set of tracklets."""
    if not path.exists():
        print(
            "Warning: no review key found, so the judgements cannot be verified "
            "against the current tracks. Re-run 'review' to make one."
        )
        return

    stored = json.loads(path.read_text(encoding="utf-8"))["fingerprint"]
    if stored != fingerprint(tracks):
        raise SystemExit(
            "The judgements were made against a different set of tracklets "
            f"(key {stored}, current {fingerprint(tracks)}). Track ids are "
            "renumbered every time refine runs, so the verdicts no longer "
            "describe these tracklets. Re-run 'review --overwrite' and judge "
            "the new contact sheets."
        )


VERDICT_CLEAN = "clean"
VERDICT_MIXED = "mixed"
VERDICT_UNSURE = "unsure"
VERDICTS = (VERDICT_CLEAN, VERDICT_MIXED, VERDICT_UNSURE)
COLUMNS = ("track_id", "verdict", "team", "player")


@dataclass(frozen=True)
class IdentityReport:
    judged: int
    judged_seconds: float
    clean_rate: float
    clean_rate_weighted: float
    mixed: int
    unsure: int
    team_accuracy: float | None
    team_compared: int
    players_named: int
    tracklets_per_player: float | None

    def format(self) -> str:
        lines = [
            f"Judged {self.judged} tracklets ({self.judged_seconds:.0f} s of track time)",
            f"  clean: {self.clean_rate * 100:.0f} percent by count, "
            f"{self.clean_rate_weighted * 100:.0f} percent by duration",
            f"  mixed: {self.mixed}   unsure: {self.unsure}",
        ]
        if self.team_accuracy is not None:
            lines.append(
                f"  team label agrees with the human on {self.team_accuracy * 100:.0f} "
                f"percent of {self.team_compared} comparable tracklets"
            )
        if self.tracklets_per_player is not None:
            lines.append(
                f"  {self.players_named} players named, "
                f"{self.tracklets_per_player:.1f} tracklets per player"
            )
        return "\n".join(lines)


def read_reviews(path: Path) -> pd.DataFrame:
    """Load human judgements, tolerating absent optional columns."""
    table = pd.read_csv(path)
    missing = {"track_id", "verdict"} - set(table.columns)
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {sorted(missing)}")

    for optional in ("team", "player"):
        if optional not in table.columns:
            table[optional] = ""

    table["verdict"] = table["verdict"].astype(str).str.strip().str.lower()
    unknown = set(table["verdict"]) - set(VERDICTS)
    if unknown:
        raise SystemExit(f"Unknown verdict(s) {sorted(unknown)}; expected one of {VERDICTS}")

    table["team"] = table["team"].fillna("").astype(str).str.strip()
    table["player"] = table["player"].fillna("").astype(str).str.strip()
    return table


def evaluate(
    reviews: pd.DataFrame,
    lengths_s: pd.Series,
    predicted_team: pd.Series | None = None,
) -> IdentityReport:
    """Identity quality against the human judgements."""
    reviews = reviews[reviews["track_id"].isin(lengths_s.index)].copy()
    reviews["seconds"] = reviews["track_id"].map(lengths_s)

    decided = reviews[reviews["verdict"] != VERDICT_UNSURE]
    clean = decided[decided["verdict"] == VERDICT_CLEAN]

    clean_rate = len(clean) / len(decided) if len(decided) else 0.0
    weighted = (
        clean["seconds"].sum() / decided["seconds"].sum() if decided["seconds"].sum() else 0.0
    )

    team_accuracy, compared = None, 0
    if predicted_team is not None:
        comparable = clean[clean["team"].isin(("A", "B", "other"))].copy()
        comparable["predicted"] = comparable["track_id"].map(predicted_team)
        comparable = comparable[comparable["predicted"].notna()]
        compared = len(comparable)
        if compared:
            team_accuracy = float((comparable["team"] == comparable["predicted"]).mean())

    named = clean[clean["player"] != ""]
    players = named["player"].nunique()
    per_player = len(named) / players if players else None

    return IdentityReport(
        judged=len(reviews),
        judged_seconds=float(reviews["seconds"].sum()),
        clean_rate=clean_rate,
        clean_rate_weighted=float(weighted),
        mixed=int((reviews["verdict"] == VERDICT_MIXED).sum()),
        unsure=int((reviews["verdict"] == VERDICT_UNSURE).sum()),
        team_accuracy=team_accuracy,
        team_compared=compared,
        players_named=players,
        tracklets_per_player=per_player,
    )
