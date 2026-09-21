"""Drawing helpers for the calibration and tracking sanity videos."""

from __future__ import annotations

import cv2
import numpy as np

from pitchmap.calib.homography import apply
from pitchmap.calib.pitch import line_segments
from pitchmap.config import PitchConfig

COLOUR_FITTED = (0, 255, 255)
COLOUR_POOR = (0, 80, 255)
MINIMAP_SCALE = 3


def line_colour(residual_m: float, warn_m: float) -> tuple[int, int, int]:
    """Colour the reprojected pitch by how much its frame's fit can be trusted."""
    if np.isfinite(residual_m) and residual_m > warn_m:
        return COLOUR_POOR
    return COLOUR_FITTED


def draw_pitch_lines(
    frame: np.ndarray,
    matrix: np.ndarray,
    pitch: PitchConfig,
    colour: tuple[int, int, int],
) -> np.ndarray:
    """Reproject the pitch model into the image through the inverse homography."""
    canvas = frame.copy()
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return canvas

    for segment in line_segments(pitch):
        pixels = apply(inverse, segment)
        if not np.all(np.isfinite(pixels)):
            continue
        points = pixels.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], isClosed=False, color=colour, thickness=2)

    return canvas


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    for row, text in enumerate(lines):
        origin = (10, 26 + row * 24)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return frame


def draw_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    colour: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw plain boxes with no ids, for raw detections."""
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    return frame


def draw_tracks(
    frame: np.ndarray,
    boxes: np.ndarray,
    track_ids: np.ndarray,
    colours: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw track boxes, optionally coloured per track id (used for teams)."""
    for (x1, y1, x2, y2), track_id in zip(boxes.astype(int), track_ids, strict=True):
        colour = (0, 255, 0) if colours is None else colours.get(int(track_id), (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            frame, str(int(track_id)), (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3
        )
        cv2.putText(
            frame, str(int(track_id)), (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1
        )
    return frame


def minimap(
    pitch: PitchConfig,
    points: np.ndarray,
    track_ids: np.ndarray,
    colours: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Top-down pitch showing where the tracked players were projected."""
    height = int(pitch.width_m * MINIMAP_SCALE)
    width = int(pitch.length_m * MINIMAP_SCALE)
    canvas = np.full((height, width, 3), (40, 90, 40), dtype=np.uint8)

    for segment in line_segments(pitch):
        pixels = (segment * MINIMAP_SCALE).astype(np.int32)
        cv2.polylines(canvas, [pixels], isClosed=False, color=(220, 220, 220), thickness=1)

    for (x_m, y_m), track_id in zip(points, track_ids, strict=True):
        if not np.isfinite([x_m, y_m]).all():
            continue
        centre = (int(x_m * MINIMAP_SCALE), int(y_m * MINIMAP_SCALE))
        dot = (0, 220, 255) if colours is None else colours.get(int(track_id), (0, 220, 255))
        cv2.circle(canvas, centre, 4, dot, -1)
        cv2.putText(
            canvas, str(int(track_id)), (centre[0] + 5, centre[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
        )

    return canvas


def inset(frame: np.ndarray, panel: np.ndarray, margin: int = 12) -> np.ndarray:
    """Place a panel in the bottom-right corner of a frame."""
    scale = min(1.0, (frame.shape[1] * 0.32) / panel.shape[1])
    resized = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)))
    y1 = frame.shape[0] - resized.shape[0] - margin
    x1 = frame.shape[1] - resized.shape[1] - margin
    frame[y1 : y1 + resized.shape[0], x1 : x1 + resized.shape[1]] = resized
    return frame


def with_minimap(
    frame: np.ndarray,
    pitch: PitchConfig,
    projected,
    colours: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Composite the pitch inset for one frame's valid projected positions."""
    if projected is None:
        return frame
    valid = projected[projected["valid"]]
    panel = minimap(
        pitch, valid[["x_m", "y_m"]].to_numpy(), valid["track_id"].to_numpy(), colours
    )
    return inset(frame, panel)


def show(window: str, canvas: np.ndarray) -> bool:
    """Display a frame in a window. Returns False when the user presses q."""
    cv2.imshow(window, canvas)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def writer(path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
