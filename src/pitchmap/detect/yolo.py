"""Ultralytics YOLO detector.

Every class the model predicts is stored. Which of them are worth tracking is
the track stage's decision, so a detector swap or a change of mind about
referees does not mean re-running detection.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from pitchmap.config import DetectConfig


class YoloDetector:
    def __init__(self, config: DetectConfig) -> None:
        self.device = config.device if torch.cuda.is_available() else "cpu"
        self.conf = config.conf_store
        self.imgsz = config.imgsz
        self.model = YOLO(config.model_path).to(self.device)

    @property
    def class_names(self) -> dict[int, str]:
        """Class id to name, as the loaded weights define it.

        Ids are model-specific, so callers resolve them to names here rather
        than hard-coding an ordering that a different checkpoint would break.
        """
        return self.model.names

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

    def describe(self) -> str:
        classes = ",".join(self.class_names[key] for key in sorted(self.class_names))
        return (
            f"yolo({self.model.model_name}, imgsz={self.imgsz}, "
            f"device={self.device}, classes=[{classes}])"
        )
