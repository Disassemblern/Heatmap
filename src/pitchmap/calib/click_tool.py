"""Interactive landmark annotation for keyframes.

The tool walks through the landmark catalogue one entry at a time, showing where
the current landmark sits on a schematic pitch, so no landmark-selection UI is
needed. Progress is saved after every keyframe, so the session can be
interrupted and resumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pitchmap.calib.homography import MIN_POINTS, fit_keyframe
from pitchmap.calib.pitch import landmarks, line_segments
from pitchmap.config import PitchConfig

WINDOW_FRAME = "keyframe (click landmark)"
WINDOW_PITCH = "pitch reference"
SCHEMATIC_SCALE = 7
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
KEY_UNDO = ord("u")
KEY_SKIP = 32
KEY_NEXT_KEYFRAME = ord("n")
KEY_PREV_KEYFRAME = ord("p")


@dataclass
class ClickState:
    point: tuple[float, float] | None = None


def draw_schematic(pitch: PitchConfig, highlight: str | None) -> np.ndarray:
    """Top-down pitch with the requested landmark highlighted."""
    height = int(pitch.width_m * SCHEMATIC_SCALE) + 40
    width = int(pitch.length_m * SCHEMATIC_SCALE) + 40
    canvas = np.full((height, width, 3), (40, 90, 40), dtype=np.uint8)

    def to_px(point: tuple[float, float]) -> tuple[int, int]:
        return (int(point[0] * SCHEMATIC_SCALE) + 20, int(point[1] * SCHEMATIC_SCALE) + 20)

    for segment in line_segments(pitch):
        points = np.array([to_px(tuple(point)) for point in segment], dtype=np.int32)
        cv2.polylines(canvas, [points], isClosed=False, color=(230, 230, 230), thickness=1)

    if highlight is not None:
        target = landmarks(pitch).get(highlight)
        if target is not None:
            cv2.circle(canvas, to_px(target), 9, (0, 220, 255), 2)
            cv2.circle(canvas, to_px(target), 2, (0, 220, 255), -1)

    return canvas


def _draw_frame(
    frame: np.ndarray,
    clicked: dict[str, tuple[float, float]],
    landmark: str,
    frame_index: int,
    position: tuple[int, int],
) -> np.ndarray:
    canvas = frame.copy()
    for name, (x, y) in clicked.items():
        cv2.drawMarker(canvas, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(
            canvas, name[:18], (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )

    index, total = position
    header = [
        f"frame {frame_index}   keyframe {index + 1}/{total}   clicked {len(clicked)}",
        f"CLICK: {landmark}",
        "space=skip  u=undo  n/p=next/prev keyframe  s=save  q=save+quit",
    ]
    for row, text in enumerate(header):
        origin = (10, 24 + row * 22)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return canvas


def annotate_keyframe(
    frame: np.ndarray,
    frame_index: int,
    pitch: PitchConfig,
    existing: dict[str, tuple[float, float]],
    position: tuple[int, int],
) -> tuple[dict[str, tuple[float, float]], str]:
    """Collect landmark clicks for one keyframe.

    Returns the clicks and the action that ended the keyframe: "next", "prev",
    "save", or "quit".
    """
    clicked = dict(existing)
    state = ClickState()
    catalogue = list(landmarks(pitch))
    order = [name for name in catalogue if name not in clicked] + [
        name for name in catalogue if name in clicked
    ]

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.point = (float(x), float(y))

    cv2.namedWindow(WINDOW_FRAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_FRAME, on_mouse)
    cv2.namedWindow(WINDOW_PITCH, cv2.WINDOW_NORMAL)

    cursor = 0
    history: list[str] = []
    while cursor < len(order):
        landmark = order[cursor]
        cv2.imshow(WINDOW_FRAME, _draw_frame(frame, clicked, landmark, frame_index, position))
        cv2.imshow(WINDOW_PITCH, draw_schematic(pitch, landmark))

        key = cv2.waitKey(20) & 0xFF
        if state.point is not None:
            clicked[landmark] = state.point
            history.append(landmark)
            state.point = None
            cursor += 1
            continue

        if key == KEY_SKIP:
            cursor += 1
        elif key == KEY_UNDO and history:
            clicked.pop(history.pop(), None)
            cursor = max(0, cursor - 1)
        elif key == KEY_NEXT_KEYFRAME:
            return clicked, "next"
        elif key == KEY_PREV_KEYFRAME:
            return clicked, "prev"
        elif key == KEY_SAVE:
            return clicked, "save"
        elif key == KEY_QUIT:
            return clicked, "quit"

    return clicked, "next"


def report_fit(
    clicked: dict[str, tuple[float, float]],
    pitch: PitchConfig,
    ransac_reproj_px: float,
) -> None:
    """Print an immediate fit check so a mis-click is caught while it is fixable."""
    if len(clicked) < MIN_POINTS:
        print(f"  only {len(clicked)} landmarks clicked, need {MIN_POINTS} to fit")
        return

    try:
        fit = fit_keyframe(clicked, pitch, ransac_reproj_px)
    except ValueError as error:
        print(f"  fit failed: {error}")
        return

    print(
        f"  fit on {fit.n_points} landmarks ({fit.n_inliers} inliers): "
        f"median error {fit.median_error_m:.2f} m, worst {fit.max_error_m:.2f} m"
    )
    worst_index = int(np.argmax(fit.errors_m))
    if fit.max_error_m > 2.0:
        print(f"  check '{fit.names[worst_index]}' - its error is high, it may be mis-clicked")


def close_windows() -> None:
    cv2.destroyWindow(WINDOW_FRAME)
    cv2.destroyWindow(WINDOW_PITCH)
