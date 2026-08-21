"""Ultralytics YOLO detector restricted to the person class."""

from __future__ import annotations

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from pitchmap.config import DetectConfig

PERSON_CLASS = 0


class YoloDetector:
    def __init__(self, config: DetectConfig) -> None:
        self.device = config.device if torch.cuda.is_available() else "cpu"
        self.conf = config.conf_store
        self.imgsz = config.imgsz
        self.model = YOLO(config.model_path).to(self.device)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            classes=[PERSON_CLASS],
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

    def describe(self) -> str:
        return f"yolo({self.model.model_name}, imgsz={self.imgsz}, device={self.device})"
