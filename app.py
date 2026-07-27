"""
app.py
Main application file
BoMeyering 2026
"""

import os
import json
import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from logging import getLogger
from fastapi import FastAPI, HTTPException, Query

from src.models import (
    DebugData,
    DetectionData,
    FullPipelineMetadata,
    FullPipelineResponse,
    HealthCheckResponse,
    InferenceRequest,
    RawInferenceData,
    RawInferenceResponse,
    SegmentationData,
)
from src.inference import SegformerInference, EfficientDetInference
from src.utils import (
    array_to_b64,
    compute_class_proportions,
    find_marker_midpoints_from_mask,
    find_quadrat_corners_from_mask,
    grayscale_image_to_b64,
    preprocess,
    warp_to_roi,
    _scale_dets
)

logger = getLogger()

load_dotenv()
PORT = int(os.getenv("PORT", 80))
MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", 1024))
SEGFORMER_MODEL_PATH = os.getenv("SEGFORMER_MODEL_PATH", "onnx/segformer_enb3_pgcviewv2.onnx")
EFFICIENTDET_MODEL_PATH = os.getenv("EFFICIENTDET_MODEL_PATH", "onnx/effdet_d2_epoch10_pgcviewv2.onnx")
CLASS_MAPPING_PATH = os.getenv("CLASS_MAPPING_PATH", "class_mapping.json")

with open(CLASS_MAPPING_PATH) as f:
    CLASS_MAPPING = json.load(f)

# Ordered class name list; index corresponds to the segformer class_idx.
SEGFORMER_CLASS_NAMES: list[str] = [
    name for name, _ in sorted(
        CLASS_MAPPING["segformer"].items(), key=lambda kv: kv[1]["class_idx"]
    )
]

# pixel value -> class name (for segmentation map interpretation)
PIXEL_CLASS_MAP: dict[int, str] = {
    v["class_idx"]: k for k, v in CLASS_MAPPING["segformer"].items()
}

# bbox class id -> class name (effdet is 1-indexed in detections)
BBOX_CLASS_MAP: dict[int, str] = {
    v["class_idx"]: k for k, v in CLASS_MAPPING["effdet"].items()
}

# EfficientDet classes are 1-indexed in detections.
MARKER_CLASS_ID  = CLASS_MAPPING["effdet"]["marker"]["class_idx"]
QUADRAT_CLASS_ID = CLASS_MAPPING["effdet"]["quadrat_corner"]["class_idx"]

# Segformer class indices for the objects used in ROI recovery.
# Set these to match your model's training class ordering.
# -1 disables recovery for that type (early-return with warning instead).
SEGFORMER_MARKER_CLASS_IDX  = int(os.getenv("SEGFORMER_MARKER_CLASS_IDX",  "-1"))
SEGFORMER_QUADRAT_CLASS_IDX = int(os.getenv("SEGFORMER_QUADRAT_CLASS_IDX", "-1"))

segformer_inference = None
efficientdet_inference = None
request_count = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global segformer_inference, efficientdet_inference
    try:
        segformer_inference = SegformerInference(SEGFORMER_MODEL_PATH)
        efficientdet_inference = EfficientDetInference(EFFICIENTDET_MODEL_PATH)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
    yield

app = FastAPI(lifespan=lifespan)

#--------Endpoints---------#
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return {"status": "healthy"}

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/stats")
async def stats():
    return {"total_requests": request_count}

@app.post("/raw_inference", response_model=RawInferenceResponse)
async def raw_infer(
    request: InferenceRequest,
    conf: float = Query(0.85, ge=0.0, le=1.0),
):
    global request_count
    img, orig_size = preprocess(request)

    seg_raw = segformer_inference.run(img)
    dets = _scale_dets(efficientdet_inference.run(img, conf), orig_size)

    response = RawInferenceResponse(
        metadata={"request_count": request_count},
        data=RawInferenceData(
            logits=array_to_b64(seg_raw[0]),
            bboxes=array_to_b64(dets),
            pixel_class_map=PIXEL_CLASS_MAP,
            bbox_class_map=BBOX_CLASS_MAP,
        ),
    )
    request_count += 1
    return response


@app.post("/full_pipeline", response_model=FullPipelineResponse)
async def full_pipeline(
    request: InferenceRequest,
    conf: float = Query(0.85, ge=0.0, le=1.0),
    marker_type: str = Query("all", pattern="^(all|marker|quadrat)$"),
    logits: bool = Query(False),
    bboxes: bool = Query(False),
    output_map: bool = Query(False),
):
    """Run both models and return per-class pixel proportions within the detected ROI.

    Detection logic:
      1. Filter all detections to those above `conf`.
      2. Sort by score descending; inspect the top-2 to determine image type
         (marker vs quadrat).  If `marker_type` is set explicitly it overrides.
      3. Take the top-4 detections of the determined type.  If fewer than 4 are
         found above `conf`, return an early response with a warning.
      4. Compute the midpoint (cx, cy) of each of the 4 bounding boxes.
      5. Use those 4 midpoints as source corners for cv2.warpPerspective,
         mapping them to the corners of a MODEL_IMAGE_SIZE × MODEL_IMAGE_SIZE canvas.
      6. Apply the same warp to the argmax segmentation class map.
      7. Count the proportion of pixels in each segformer class across the
         warped ROI and return as a vector.
    """
    global request_count

    print(f"Received request #{request_count + 1}: conf={conf}, marker_type={marker_type}, logits={logits}, bboxes={bboxes}, output_map={output_map}")

    img, orig_size = preprocess(request)
    seg_raw = segformer_inference.run(img)
    # Scale detections from model space (MODEL_IMAGE_SIZE²) to original image space
    dets = _scale_dets(efficientdet_inference.run(img, conf), orig_size)  # [N, 6]

    # Upsample logits to original image size before argmax for the highest-fidelity class map
    class_map = segformer_inference.postprocess(seg_raw, target_size=orig_size)  # [H, W] uint8

    img_name = (request.metadata or {}).get("img_name")

    orig_h, orig_w = orig_size

    def _response(image_type, num_det, region_type, roi_bboxes, detection_scores,
                  num_inferred, inferred_pts, class_proportions,
                  class_map_arr, dets_arr, seg_raw_arr, warning=None):
        return FullPipelineResponse(
            metadata=FullPipelineMetadata(
                request_count=request_count,
                img_name=img_name,
                input_size=max(orig_h, orig_w),
                image_type=image_type,
            ),
            segmentation=SegmentationData(
                output_map=grayscale_image_to_b64(class_map_arr),
                pixel_class_map=PIXEL_CLASS_MAP,
                class_proportions=class_proportions,
            ),
            detection=DetectionData(
                num_detections=num_det,
                region_type=region_type,
                roi_bboxes=roi_bboxes,
                detection_scores=detection_scores,
                num_inferred=num_inferred,
                inferred_points=inferred_pts,
                bbox_class_map=BBOX_CLASS_MAP,
            ),
            debug=DebugData(
                bboxes=array_to_b64(dets_arr) if bboxes else None,
                logits=array_to_b64(seg_raw_arr[0]) if logits else None,
            ),
            errors={"warning": warning} if warning else None,
        )

    # --- No detections at all: fall back to class proportions over the full raw image ---
    if dets.shape[0] == 0:
        return _response(
            "unknown", 0, "full_image", [], [], 0, [],
            compute_class_proportions(class_map, SEGFORMER_CLASS_NAMES),
            class_map, dets, seg_raw,
            warning="No detections above confidence threshold; returning class proportions over the full image.",
        )

    # Sort detections by score descending (col 4 = score)
    sorted_dets = dets[np.argsort(dets[:, 4])[::-1]]

    # --- Determine image type ---
    top2_classes = sorted_dets[:min(2, len(sorted_dets)), 5].astype(int)

    if marker_type == "marker":
        image_type = "marker"
        target_cls_id = MARKER_CLASS_ID
    elif marker_type == "quadrat":
        image_type = "quadrat"
        target_cls_id = QUADRAT_CLASS_ID
    else:
        # "all": let top-2 detections decide
        if np.all(top2_classes == MARKER_CLASS_ID):
            image_type = "marker"
            target_cls_id = MARKER_CLASS_ID
        elif np.all(top2_classes == QUADRAT_CLASS_ID):
            image_type = "quadrat"
            target_cls_id = QUADRAT_CLASS_ID
        else:
            # Mixed top-2: fall back to majority vote across top-4
            top4_classes = sorted_dets[:min(4, len(sorted_dets)), 5].astype(int)
            n_marker  = int(np.sum(top4_classes == MARKER_CLASS_ID))
            n_quadrat = int(np.sum(top4_classes == QUADRAT_CLASS_ID))
            if n_marker >= n_quadrat:
                image_type = "marker"
                target_cls_id = MARKER_CLASS_ID
            else:
                image_type = "quadrat"
                target_cls_id = QUADRAT_CLASS_ID

    # --- Get top 4 of the determined type ---
    type_dets = sorted_dets[sorted_dets[:, 5].astype(int) == target_cls_id]
    top4 = type_dets[:4]
    n_detected = len(top4)
    n_missing  = 4 - n_detected

    # --- Compute midpoints for whatever was detected ---
    if n_detected > 0:
        detected_midpoints = np.stack([
            (top4[:, 0] + top4[:, 2]) / 2,
            (top4[:, 1] + top4[:, 3]) / 2,
        ], axis=1).astype(np.float32)            # [n_detected, 2]
    else:
        detected_midpoints = np.empty((0, 2), dtype=np.float32)

    # --- Recover missing positions from the segmentation map ---
    n_recovered = 0
    if n_missing > 0:
        if image_type == "marker":
            recovered = find_marker_midpoints_from_mask(
                class_map, SEGFORMER_MARKER_CLASS_IDX,
                detected_midpoints, n_missing,
                detected_bboxes=top4[:, :4],
            )
        else:
            recovered = find_quadrat_corners_from_mask(
                class_map, SEGFORMER_QUADRAT_CLASS_IDX,
                n_missing,
                detected_bboxes=top4[:, :4],
            )

        n_recovered = len(recovered)
        if n_recovered < n_missing:
            partial_roi_bboxes = top4[:, :4].tolist()
            partial_roi_bboxes += [[float(cx), float(cy), float(cx), float(cy)] for cx, cy in recovered]
            request_count += 1
            return _response(
                image_type, n_detected, "full_image",
                partial_roi_bboxes,
                [float(s) for s in top4[:, 4]],
                n_recovered,
                [[float(x), float(y)] for x, y in recovered],
                compute_class_proportions(class_map, SEGFORMER_CLASS_NAMES),
                class_map, top4, seg_raw,
                warning=(
                    f"Need 4 '{image_type}' positions; detected {n_detected}, "
                    f"recovered {n_recovered}/{n_missing} from segmentation mask; "
                    "returning class proportions over the full image."
                ),
            )

        midpoints = (
            np.concatenate([detected_midpoints, recovered], axis=0)
            if n_detected > 0
            else recovered
        )
    else:
        midpoints = detected_midpoints

    # --- Perspective warp of the class map ---
    # order_points uses Hungarian matching against image corners to ensure
    # correct TL/TR/BR/BL correspondence for cv2.getPerspectiveTransform.
    warped_class_map = warp_to_roi(class_map, midpoints, MODEL_IMAGE_SIZE)

    # --- Compute per-class proportions in the warped ROI ---
    proportions = compute_class_proportions(warped_class_map, SEGFORMER_CLASS_NAMES)

    inferred_points = midpoints[n_detected:] if n_recovered > 0 else np.empty((0, 2), dtype=np.float32)

    roi_bboxes = top4[:, :4].tolist()
    roi_bboxes += [[float(cx), float(cy), float(cx), float(cy)] for cx, cy in inferred_points]

    request_count += 1

    return _response(
        image_type, n_detected, image_type,
        roi_bboxes,
        [float(s) for s in top4[:, 4]],
        n_recovered,
        [[float(x), float(y)] for x, y in inferred_points],
        proportions,
        class_map, top4, seg_raw,
    )


if __name__ == "__main__":
    logger.info(f"Starting PGCView v2 server on port {PORT}")
    uvicorn.run("app:app", host="0.0.0.0", port=PORT)
