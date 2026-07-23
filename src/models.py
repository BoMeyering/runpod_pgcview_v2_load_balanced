"""
src/models.py
Pydantic request and response models
BoMeyering 2026
"""

from typing import Optional
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    b64_str: str
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# /raw_inference
# ---------------------------------------------------------------------------

class RawInferenceData(BaseModel):
    logits: str                        # base64 encoded [1, C, H, W] float32 array
    bboxes: str                        # base64 encoded [N, 6] float32 array
    pixel_class_map: dict[int, str]
    bbox_class_map: dict[int, str]


class RawInferenceResponse(BaseModel):
    metadata: dict
    data: RawInferenceData
    errors: Optional[dict] = None


# ---------------------------------------------------------------------------
# /full_pipeline
# ---------------------------------------------------------------------------

class FullPipelineMetadata(BaseModel):
    request_count: int
    img_name: Optional[str] = None
    input_size: int
    image_type: str                    # "marker", "quadrat", or "unknown"


class SegmentationData(BaseModel):
    output_map: str                    # base64 encoded full-image segmentation PNG
    pixel_class_map: dict[int, str]   # pixel value -> class name
    class_proportions: dict[str, float]


class DetectionData(BaseModel):
    num_detections: int
    region_type: Optional[str] = None  # "marker", "quadrat", or "full_image"
    roi_bboxes: list[list[float]] = [] # [[x1,y1,x2,y2], ...] detected + imputed point-boxes
    detection_scores: list[float] = [] # confidence scores of detected bboxes
    num_inferred: int = 0              # positions recovered from segmentation mask
    inferred_points: list[list[float]] = []
    bbox_class_map: dict[int, str]    # class id -> class name


class DebugData(BaseModel):
    bboxes: Optional[str] = None       # base64 encoded raw [N, 6] det array
    logits: Optional[str] = None       # base64 encoded raw [1, C, H, W] logits


class FullPipelineResponse(BaseModel):
    metadata: FullPipelineMetadata
    segmentation: SegmentationData
    detection: DetectionData
    debug: DebugData
    errors: Optional[dict] = None


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthCheckResponse(BaseModel):
    status: str
