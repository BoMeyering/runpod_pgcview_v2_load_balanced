"""
test_full_pipeline.py
Send images in a directory to /full_pipeline, decode the responses, and render an overlay of
the returned class map + marker/quadrat-corner bounding boxes on the original
image for visual inspection.
"""

import argparse
import base64
import json
import os

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

print("RUNPOD_API_KEY:", os.getenv("RUNPOD_API_KEY", "(not set)"))

INFERENCE_SIZE = 1024

CLASS_MAPPING_PATH = os.getenv("CLASS_MAPPING_PATH", "class_mapping.json")
with open(CLASS_MAPPING_PATH) as f:
    CLASS_MAPPING = json.load(f)

# Set up the inference palette for the SegFormer predictions
# rgb -> BGR for cv2, indexed by segformer class_idx.
SEGFORMER_PALETTE = np.zeros((max(v["class_idx"] for v in CLASS_MAPPING["segformer"].values()) + 1, 3), dtype=np.uint8)
for _name, _info in CLASS_MAPPING["segformer"].items():
    SEGFORMER_PALETTE[_info["class_idx"]] = _info["rgb"][::-1]

# EfficientDet: name/color lookups keyed by class_idx.
CLASS_ID_NAMES = {info["class_idx"]: name for name, info in CLASS_MAPPING["effdet"].items()}
BBOX_COLORS    = {info["class_idx"]: tuple(info["rgb"][::-1]) for info in CLASS_MAPPING["effdet"].values()}
EFFDET_NAME_COLORS = {name: tuple(info["rgb"][::-1]) for name, info in CLASS_MAPPING["effdet"].items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Test the /full_pipeline endpoint and visualize the result.")
    parser.add_argument(
        "--image-dir", 
        default="test_images", 
        help="Path to the input image dir."
    )
    parser.add_argument(
        "--url", 
        default="https://cds0121q75mfue.api.runpod.ai", 
        help="Base URL of the running server."
    )
    parser.add_argument(
        "--api-key", 
        default=os.getenv("RUNPOD_API_KEY"), 
        help="RunPod API key (or set RUNPOD_API_KEY env var)."
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.85, 
        help="Detection confidence threshold."
    )
    parser.add_argument(
        "--marker-type", 
        default="all", 
        choices=["all", "marker", "quadrat"]
    )
    parser.add_argument(
        "--output-dir", 
        default="test_output", 
        help="Path to write the overlay image."
    )

    return parser.parse_args()


def decode_class_map(b64_str: str) -> np.ndarray:
    """
    Decode a base64-encoded PNG representing a class map into a grayscale numpy array.
    """
    png_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(png_bytes, dtype=np.uint8)

    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def decode_bboxes(b64_str: str) -> np.ndarray:
    """
    Decode a base64-encoded [N, 6] float32 array of detection boxes
    (x1, y1, x2, y2, score, class_id)
    """
    raw = base64.b64decode(b64_str)

    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 6)


def scale_xyxy(boxes, sx: float, sy: float) -> np.ndarray:
    """
    Scale [..., x1, y1, x2, y2] boxes from inference space back to the original image size.
    """
    boxes = np.array(boxes, dtype=np.float32, copy=True)
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    return boxes


def scale_xy(points, sx: float, sy: float) -> np.ndarray:
    """
    Scale [..., x, y] points from inference space back to the original image size.
    """
    points = np.array(points, dtype=np.float32, copy=True)
    points[:, 0] *= sx
    points[:, 1] *= sy

    return points


def overlay_class_map(image: np.ndarray, class_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a class map on top of the original image with transparency.
    """
    resized_map = cv2.resize(class_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_map = SEGFORMER_PALETTE[resized_map]
    blended = image.copy()
    fg_mask = resized_map != 0
    blended[fg_mask] = cv2.addWeighted(image, 1 - alpha, color_map, alpha, 0)[fg_mask]

    return blended


def draw_bboxes(image: np.ndarray, dets: np.ndarray) -> np.ndarray:
    """
    Draw detection boxes on the image with class labels and confidence scores.
    """
    out = image.copy()
    for x1, y1, x2, y2, score, cls_id in dets:
        cls_id = int(cls_id)
        color = BBOX_COLORS.get(cls_id, (0, 255, 255))
        label = f"{CLASS_ID_NAMES.get(cls_id, str(cls_id))} {score:.2f}"
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, label, (int(x1), max(int(y1) - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
    return out


def _dashed_rect(
        image: np.ndarray, 
        pt1: tuple, 
        pt2: tuple, 
        color: tuple, 
        thickness: int=2, 
        dash_len: int=10
    ):
    """
    Draw a dashed rectangle so recovered points read as distinct from solid detection boxes.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    for xa, ya, xb, yb in [(x1, y1, x2, y1), (x1, y2, x2, y2), (x1, y1, x1, y2), (x2, y1, x2, y2)]:
        length = max(int(np.hypot(xb - xa, yb - ya)), 1)
        for i in range(0, length, dash_len * 2):
            t0, t1 = i / length, min((i + dash_len) / length, 1.0)
            start = (int(xa + (xb - xa) * t0), int(ya + (yb - ya) * t0))
            end   = (int(xa + (xb - xa) * t1), int(ya + (yb - ya) * t1))
            cv2.line(image, start, end, color, thickness)


def draw_inferred_points(
        image: np.ndarray, 
        points: list, 
        effdet_name: str, 
        box_frac: float=0.02
    ) -> np.ndarray:
    """
    Draw a dashed box around each position recovered from the segmentation mask.
    """

    out = image.copy()
    color = EFFDET_NAME_COLORS.get(effdet_name, (0, 255, 255))
    half = max(int(max(image.shape[:2]) * box_frac), 15)
    for x, y in points:
        x, y = int(x), int(y)
        _dashed_rect(out, (x - half, y - half), (x + half, y + half), color, thickness=2)
        cv2.putText(out, f"{effdet_name} (inferred)", (x - half, max(y - half - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
    return out


def draw_roi_polygon(
        image: np.ndarray, 
        roi_bboxes: list, 
        color: tuple=(0, 255, 255)
    ) -> np.ndarray:
    """
    Connect the midpoints of the 4 ROI boxes (detected + inferred) to show the warp quadrilateral.
    """

    midpoints = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in roi_bboxes], dtype=np.float32)
    centroid = midpoints.mean(axis=0)
    angles = np.arctan2(midpoints[:, 1] - centroid[1], midpoints[:, 0] - centroid[0])
    ordered = midpoints[np.argsort(angles)].astype(np.int32)

    out = image.copy()
    cv2.polylines(out, [ordered.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(out, "ROI", tuple(ordered[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return out


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    headers = {"Authorization": f"Bearer {args.api_key}"}
    params  = {"conf": args.conf, "marker_type": args.marker_type, "bboxes": True}

    # Retry the request a few times in case the server is still starting up
    for attempt in range(5):
        try:
            ping_response = requests.get(f"{args.url}/ping", headers=headers)
            if ping_response.ok:
                break
            else:
                print(f"Ping attempt {attempt + 1} failed: {ping_response.status_code} {ping_response.text}")
        except requests.RequestException as e:
            print(f"Ping attempt {attempt + 1} raised an exception: {e}")
        if attempt < 4:
            print("Waiting 5 seconds before retrying...")
            import time
            time.sleep(5)

    for filename in os.listdir(args.image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(args.image_dir, filename)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        image_resize= cv2.resize(image, (INFERENCE_SIZE, INFERENCE_SIZE))
        if image_resize is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        _, buffer = cv2.imencode(".jpg", image_resize, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_str = base64.b64encode(buffer).decode("utf-8")

        payload = {"b64_str": b64_str, "metadata": {"img_name": image_path}}

        if not args.api_key:
            raise ValueError("No API key provided. Pass --api-key or set the RUNPOD_API_KEY environment variable.")
        
        response = requests.post(f"{args.url}/full_pipeline", json=payload, params=params, headers=headers)
        print(f"status: {response.status_code}")
        if not response.ok:
            print(f"error response: {response.text or '(empty body)'}")
            response.raise_for_status()
        body = response.json()

        print(body)

        meta      = body["metadata"]
        seg       = body["segmentation"]
        det       = body["detection"]
        debug     = body["debug"]

        print(f"image_type:     {meta['image_type']}")
        print(f"num_detections: {det['num_detections']}")
        print(f"region_type:    {det['region_type']}")
        print(f"num_inferred:   {det['num_inferred']}")
        print(f"roi_bboxes:     {det['roi_bboxes']}")
        if body.get("errors"):
            print(f"warning: {body['errors'].get('warning')}")
        print("class_proportions:")
        for name, prop in seg["class_proportions"].items():
            print(f"  {name:<15} {prop:.4f}")

        overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Server received the image resized to INFERENCE_SIZE x INFERENCE_SIZE, so all
        # coordinates in the response are in that space; scale them back up to the
        # original image for rendering (the class map is resized separately below).
        sx = overlay.shape[1] / INFERENCE_SIZE
        sy = overlay.shape[0] / INFERENCE_SIZE

        # output_map is always present in the new structure
        class_map = decode_class_map(seg["output_map"])
        overlay = overlay_class_map(overlay, class_map)

        # raw detection array (only present when bboxes=True was sent)
        dets = decode_bboxes(debug["bboxes"]) if debug.get("bboxes") else np.empty((0, 6), dtype=np.float32)
        if len(dets):
            overlay = draw_bboxes(overlay, scale_xyxy(dets, sx, sy))

        if det.get("inferred_points"):
            effdet_name = "marker" if meta["image_type"] == "marker" else "quadrat_corner"
            overlay = draw_inferred_points(overlay, scale_xy(det["inferred_points"], sx, sy), effdet_name)

        # Draw the ROI polygon when we have a real ROI (not a full_image fallback)
        if det["region_type"] in ("marker", "quadrat") and len(det["roi_bboxes"]) == 4:
            overlay = draw_roi_polygon(overlay, scale_xyxy(det["roi_bboxes"], sx, sy))

        output_path = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_overlay.png")
        cv2.imwrite(output_path, overlay)
        print(f"wrote overlay to {output_path}")

if __name__ == "__main__":
    main()
