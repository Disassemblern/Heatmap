"""MOT Challenge format, the interchange used by external tracking tools.

Columns are frame,id,left,top,width,height,conf,-1,-1,-1 with 1-based frame
indices. Conversion to and from the pipeline's 0-based indices happens here and
nowhere else.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ("frame", "id", "left", "top", "width", "height", "conf", "x", "y", "z")


def write_mot(tracks: pd.DataFrame, path: Path | str) -> None:
    """Write a tracks frame (0-based, xyxy) as MOT text (1-based, xywh)."""
    rows = pd.DataFrame(
        {
            "frame": tracks["frame"].astype(int) + 1,
            "id": tracks["track_id"].astype(int),
            "left": tracks["x1"],
            "top": tracks["y1"],
            "width": tracks["x2"] - tracks["x1"],
            "height": tracks["y2"] - tracks["y1"],
            "conf": tracks["conf"],
            "x": -1,
            "y": -1,
            "z": -1,
        }
    ).sort_values(["frame", "id"])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, header=False, index=False, float_format="%.2f")


def read_mot(path: Path | str) -> pd.DataFrame:
    """Read MOT text (1-based, xywh) into a tracks frame (0-based, xyxy)."""
    raw = pd.read_csv(path, header=None, names=list(COLUMNS))
    return pd.DataFrame(
        {
            "frame": raw["frame"].astype(int) - 1,
            "track_id": raw["id"].astype(int),
            "x1": raw["left"],
            "y1": raw["top"],
            "x2": raw["left"] + raw["width"],
            "y2": raw["top"] + raw["height"],
            "conf": raw["conf"],
        }
    )
