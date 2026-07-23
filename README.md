# PGCView V2 — Inference API

FastAPI inference server for the PGCView V2 pipeline. Runs a Segformer semantic segmentation model and an EfficientDet object detection model to measure per-class vegetation cover within a detected quadrat or marker ROI.

---

## How it works

1. A base64-encoded PNG image is sent to an endpoint.
2. The image is decoded, normalized, and passed to both ONNX models simultaneously.
3. **Segformer** produces a per-pixel class map (12 classes: soil, pgc, weed, corn, etc.).
4. **EfficientDet** detects marker posts or quadrat corner flags above a configurable confidence threshold.
5. The top 4 detections of the dominant type are used to compute a perspective warp, mapping the ROI to a fixed canvas.
6. Per-class pixel proportions are computed over the warped ROI.
7. If fewer than 4 detections are found, the server attempts to recover missing positions from the segmentation mask before falling back to full-image proportions.

---

## Setup

Copy `env.example` to `.env` and fill in model paths:

```bash
cp env.example .env
```

```env
PORT=8180
SEGFORMER_MODEL_PATH=onnx/segformer.onnx
EFFICIENTDET_MODEL_PATH=onnx/efficientdet.onnx
MODEL_IMAGE_SIZE=1024
EFFDET_ARCHITECTURE=efficientdet_d2
CLASS_MAPPING_PATH=class_mapping.json

# Set to the segformer class index for markers/quadrat corners to enable
# mask-based recovery of missing ROI corners. -1 disables recovery.
SEGFORMER_MARKER_CLASS_IDX=-1
SEGFORMER_QUADRAT_CLASS_IDX=-1
```

Start the server:

```bash
python app.py
```

---

## Endpoints

### `GET /health`

Health check required by Runpod to monitor worker status.

**Response**
```json
{ "status": "healthy" }
```

---

### `GET /stats`

Returns a running count of requests processed since startup.

**Response**
```json
{ "total_requests": 42 }
```

---

### `POST /raw_inference`

Returns the raw model outputs without any postprocessing. Useful for debugging or client-side postprocessing.

**Query params**

| Param  | Type  | Default | Description                        |
|--------|-------|---------|------------------------------------|
| `conf` | float | `0.85`  | Detection confidence threshold     |

**Request body**
```json
{
  "b64_str": "<base64 encoded PNG>",
  "metadata": { "img_name": "optional_filename.png" }
}
```

**Response**
```json
{
  "metadata": { "request_count": 1 },
  "data": {
    "logits": "<base64 encoded float32 array [1, C, H, W]>",
    "bboxes": "<base64 encoded float32 array [N, 6]>",
    "pixel_class_map": { "0": "soil", "1": "stover", "2": "pgc" },
    "bbox_class_map":  { "1": "quadrat_corner", "2": "marker" }
  },
  "errors": null
}
```

The `bboxes` array columns are `[x1, y1, x2, y2, score, class_id]`. Class IDs are 1-indexed and map to `bbox_class_map`.

---

### `POST /full_pipeline`

Runs the complete pipeline and returns per-class vegetation cover proportions within the detected ROI.

**Query params**

| Param         | Type    | Default | Description                                                               |
|---------------|---------|---------|---------------------------------------------------------------------------|
| `conf`        | float   | `0.85`  | Detection confidence threshold                                            |
| `marker_type` | string  | `"all"` | Force image type: `"all"` (auto-detect), `"marker"`, or `"quadrat"`      |
| `bboxes`      | bool    | `false` | Include raw detection array in `debug.bboxes`                            |
| `logits`      | bool    | `false` | Include raw segformer logits in `debug.logits`                           |

**Request body**
```json
{
  "b64_str": "<base64 encoded PNG>",
  "metadata": { "img_name": "optional_filename.png" }
}
```

**Response**
```json
{
  "metadata": {
    "request_count": 5,
    "img_name": "field_01.png",
    "input_size": 1024,
    "image_type": "marker"
  },
  "segmentation": {
    "output_map": "<base64 encoded grayscale PNG>",
    "pixel_class_map": { "0": "soil", "1": "stover", "2": "pgc", "...": "..." },
    "class_proportions": { "soil": 0.12, "pgc": 0.45, "weed": 0.03, "...": 0.0 }
  },
  "detection": {
    "num_detections": 4,
    "region_type": "marker",
    "roi_bboxes": [[x1, y1, x2, y2], "..."],
    "detection_scores": [0.97, 0.95, 0.91, 0.88],
    "num_inferred": 0,
    "inferred_points": [],
    "bbox_class_map": { "1": "quadrat_corner", "2": "marker" }
  },
  "debug": {
    "bboxes": "<base64 encoded float32 array [N, 6]> or null",
    "logits": "<base64 encoded float32 array [1, C, H, W]> or null"
  },
  "errors": null
}
```

**Field notes**

- `segmentation.output_map` — always returned; grayscale PNG where each pixel value is a segformer class index (use `pixel_class_map` to decode).
- `segmentation.class_proportions` — proportion of pixels in each class within the warped ROI. Sums to 1.0 across all classes.
- `detection.roi_bboxes` — up to 4 boxes `[x1, y1, x2, y2]` in original image coordinates. Detected boxes are full bounding boxes; imputed boxes (recovered from the segmentation mask) are point-boxes where `x1 == x2` and `y1 == y2`.
- `detection.num_inferred` — number of ROI corners recovered from the segmentation mask (because fewer than 4 were detected above `conf`).
- `detection.region_type` — `"marker"` or `"quadrat"` when a valid ROI was found; `"full_image"` when the pipeline fell back to whole-image proportions.
- `errors.warning` — populated when the pipeline falls back (e.g. too few detections, insufficient mask recovery).

---

## Decoding outputs

**Segmentation map (Python)**
```python
import base64, numpy as np, cv2

png_bytes = base64.b64decode(response["segmentation"]["output_map"])
class_map = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
# class_map[y, x] -> integer class index
```

**Raw bbox / logits arrays (Python)**
```python
import base64, numpy as np

bboxes = np.frombuffer(base64.b64decode(response["debug"]["bboxes"]), dtype=np.float32).reshape(-1, 6)
# columns: [x1, y1, x2, y2, score, class_id]

logits = np.frombuffer(base64.b64decode(response["debug"]["logits"]), dtype=np.float32).reshape(1, -1, 1024, 1024)
```

---

## Class mapping

`class_mapping.json` is the single source of truth for class names, indices, and overlay colors for both models.

```json
{
  "segformer": {
    "soil":    { "class_idx": 0, "rgb": [55, 52, 25] },
    "pgc":     { "class_idx": 2, "rgb": [6, 123, 2] },
    "..."
  },
  "effdet": {
    "quadrat_corner": { "class_idx": 1, "rgb": [20, 20, 20] },
    "marker":         { "class_idx": 2, "rgb": [0, 0, 255] }
  }
}
```
