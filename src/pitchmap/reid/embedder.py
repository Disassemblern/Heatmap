"""OSNet appearance embeddings for player crops.

Uses the vendored OSNet architecture with the sports-domain checkpoint, so
importing this module does not pull the torchreid package (whose import chain
drags in its training stack).

Outputs are L2-normalised here on purpose: the Deep-EIoU track state normalises
feature rows in place, and pre-normalised rows make that mutation a numerical
no-op, so a cached copy of the embeddings is identical whether it was saved
before or after the tracker consumed them.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from pitchmap.config import ReidConfig
from pitchmap.vendor.osnet import osnet_x1_0

CROP_HEIGHT = 256
CROP_WIDTH = 128
# torchreid FeatureExtractor convention: RGB in [0, 1], ImageNet statistics.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EMBEDDING_DIM = 512


class ReidEmbedder:
    def __init__(self, config: ReidConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        checkpoint = torch.load(config.model_path, map_location="cpu", weights_only=False)
        state = {
            key.replace("module.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }
        model = osnet_x1_0(num_classes=config.num_classes, pretrained=False)
        model.load_state_dict(state)
        self.model = model.eval().to(self.device)

    def embed(self, frame: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Embed one frame's crops. Returns (N, 512) float32, L2-normalised."""
        if len(boxes_xyxy) == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        batch = self._preprocess(frame, boxes_xyxy)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(batch), self.config.batch):
                chunk = batch[start : start + self.config.batch].to(self.device)
                with torch.autocast(self.device.type, enabled=self.config.fp16):
                    features = self.model(chunk)
                outputs.append(features.float().cpu())

        embeddings = torch.cat(outputs).numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def _preprocess(self, frame: np.ndarray, boxes_xyxy: np.ndarray) -> torch.Tensor:
        height, width = frame.shape[:2]
        crops = np.empty((len(boxes_xyxy), CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.float32)

        for index, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            left = int(np.clip(x1, 0, width - 2))
            top = int(np.clip(y1, 0, height - 2))
            right = int(np.clip(x2, left + 2, width))
            bottom = int(np.clip(y2, top + 2, height))

            crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            resized = cv2.resize(crop, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_LINEAR)
            crops[index] = (resized.astype(np.float32) / 255.0 - MEAN) / STD

        return torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()

    def describe(self) -> str:
        return (
            f"osnet-sports({self.config.model_path}, batch={self.config.batch}, "
            f"fp16={self.config.fp16}, device={self.config.device})"
        )
