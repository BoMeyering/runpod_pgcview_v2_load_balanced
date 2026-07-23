"""
inference.py
Inference logic for the PGCView V2 model inference API
BoMeyering 2026
Oxbow Solutions, LLC
"""

import os
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from onnxruntime import InferenceSession
from fastapi import HTTPException

from .utils import b64_to_image
from .postprocess import get_anchor_boxes, postprocess_detections
from dotenv import load_dotenv

load_dotenv()

EFFDET_ARCHITECTURE = os.getenv("EFFDET_ARCHITECTURE", "tf_efficientdet_d2")
MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", 1024))


class ModelInference(ABC):
    @abstractmethod
    def __init__(self, model_path: str):
        self.session = InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

    @abstractmethod
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def postprocess(self, outputs):
        ...

    @abstractmethod
    def run(self, img: np.ndarray):
        ...


class SegformerInference(ModelInference):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Image is already normalized and batched by utils.preprocess(); pass through.
        return img

    def postprocess(self, outputs, target_size: Optional[tuple[int, int]] = None) -> np.ndarray:
        """Interpolate logits to target_size and return a uint8 argmax class map [H, W].

        Upsampling is done on the raw logits (before argmax) via bilinear
        interpolation so that the resulting class boundaries are smoother and
        more faithful to the continuous probability surface than nearest-neighbour
        upsampling of a discrete class map would be.

        Args:
            outputs:     Raw ONNX output list; first element is [1, C, H', W'].
            target_size: (H, W) to interpolate to.  Defaults to MODEL_IMAGE_SIZE².
        """
        raw = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        logits = torch.from_numpy(raw).float()  # [1, C, H', W'] or [C, H', W']
        if logits.dim() == 3:
            logits = logits.unsqueeze(0)
        target = target_size if target_size is not None else (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
        _, _, h, w = logits.shape
        if (h, w) != target:
            logits = F.interpolate(logits, size=target, mode='bilinear', align_corners=False)
        class_map = torch.argmax(logits.squeeze(0), dim=0).numpy()
        
        return class_map.astype(np.uint8)

    def run(self, img: np.ndarray):
        """Run the segformer ONNX model and return raw output list."""
        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: img})
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Segformer model inference failed: {e}"
            )
        return outputs


class EfficientDetInference(ModelInference):
    def __init__(self, model_path: str, max_det: int = 100, nms_iou: float = 0.5, score_threshold: float = 0.85):
        super().__init__(model_path)
        self.max_det = max_det
        self.nms_iou = nms_iou
        self.score_threshold = score_threshold

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Image is already normalized and batched by utils.preprocess(); pass through.
        return img

    def postprocess(self, outputs, score_threshold: Optional[float] = None) -> np.ndarray:
        """Decode ONNX outputs to an [N, 6] float32 array of [x1, y1, x2, y2, score, class].

        Rows with class == 0 (zero-padded empties) are stripped. Class is 1-indexed.
        """
        threshold = score_threshold if score_threshold is not None else self.score_threshold
        try:
            cls_np, box_np = outputs
            anchors = get_anchor_boxes(EFFDET_ARCHITECTURE, MODEL_IMAGE_SIZE)
            dets = postprocess_detections(
                torch.from_numpy(cls_np),
                torch.from_numpy(box_np),
                anchors,
                max_detection_points=5000,
                max_det_per_image=self.max_det,
                nms_iou_threshold=self.nms_iou,
                score_threshold=threshold,
            )  # [1, max_det, 6]
            dets_img = dets[0]
            valid = dets_img[:, 5] > 0  # strip zero-padded rows
            dets_img = dets_img[valid].clone()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"EfficientDet postprocessing failed: {e}"
            )
        return dets_img.cpu().numpy()

    def run(self, img: np.ndarray, score_threshold: Optional[float] = None) -> np.ndarray:
        """Run ONNX inference and postprocess; returns [N, 6] float32 detections."""
        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: img})
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"EfficientDet ONNX inference failed: {e}"
            )
        return self.postprocess(outputs, score_threshold)
