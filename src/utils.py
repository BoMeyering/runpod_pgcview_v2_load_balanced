"""
utils.py
Utility functions for the PGCView V2 model inference API
BoMeyering 2026
Oxbow Solutions, LLC
"""

import os
import base64
import cv2
from dotenv import load_dotenv
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from logging import getLogger
from fastapi import HTTPException
from .models import InferenceRequest

logger = getLogger()

load_dotenv()
MEANS = np.array(os.getenv("MEANS", "0.485,0.456,0.406").split(","), dtype=np.float32)
STD = np.array(os.getenv("STDS", "0.229,0.224,0.225").split(","), dtype=np.float32)
MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", 1024))


def b64_to_image(b64_str: str) -> np.ndarray:
    """
    Decode a base64 PNG string to an HWC BGR numpy array.
    
    Parameters:
    -----------
    b64_str : str
        A base64 encoded png string

    Returns:
    --------
    img : np.ndarray
        A Numpy array image in H, W, C order with type uint8

    Raises:
    -------
    HTTPException : HTTP 422
        Unprocessable content in the request
    """
    try:
        bytes_data = base64.b64decode(b64_str)
        img_array = np.frombuffer(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={
                "stage": "image_decoding",
                "error": f"Failed to decode base64 string to image: {str(e)}"
            }
        )
    return img


def normalize_image(
        img: np.ndarray, 
        means: np.ndarray, 
        std: np.ndarray
    ) -> np.ndarray:
    """
    Normalize HWC image to CHW float32 using ImageNet mean/std.
    
    Constrain pixels between min 0 and max 255.
    Subtract the channel mean and divide by the standard deviation.

    Parameters:
    -----------
    img : np.ndarray
        A Numpy array image of shape H, W, C
    means : np.ndarray
        A Numpy array of shape C representing the channel mean values (float32)
    std : np.ndarray
        A Numpy array of shape C representing the channel std values (float32)

    Returns:
    --------
    normalized_img : np.ndarray
        A normalized image array in the format C, H, W
    """
    img = img.astype(np.float32) / 255.0
    img = (img - means.astype(np.float32)) / std.astype(np.float32)

    return np.transpose(img, (2, 0, 1))


def grayscale_image_to_b64(img: np.ndarray) -> str:
    """
    Encode a HW uint8 array as a base64 PNG string.
    
    Parameters:
    -----------
    img : np.ndarray
        A grayscale image of shape H, W

    Raises:
    -------
    HTTPException : HTTP 500
        Internal Server Error

    Returns:
    --------
    b64_str : str
        A base64 string representing the encoded classification map
    """
    img = img.astype(np.uint8)
    status, buffer = cv2.imencode('.png', img)
    if not status:
        raise HTTPException(
            status_code=500,
            detail={
                "stage": "image_encoding",
                "error": "Failed to encode the image to PNG format."
            }
        )
    
    return base64.b64encode(buffer).decode('utf-8')


def array_to_b64(arr: np.ndarray) -> str:
    """
    Encode a numerical data numpy array to a base64 string (raw bytes).

    Parameters:
    -----------
    arr : np.ndarray
        A numpy array of values to serialize
    
    Returns:
    --------
    b64_arr : str
        A base64 encoded string of the array bytes
    """
    return base64.b64encode(arr.tobytes()).decode('utf-8')


def preprocess(request: InferenceRequest) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Decode, resize, and normalize the base64 image from the inference request.

    Resizes the decoded image to (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE) before
    normalization so that both ONNX models always receive a square input.

    Parameters:
    -----------
    request : InferenceRequest
        The inference request with keys 'b64_str' and 'metadata'    

    Returns:
    --------
    img : np.ndarray
        A float32 Numpy array of shape [1, C, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE].
    orig_size : Tuple
        A tuple (H, W) of the decoded image before resizing. Use this to
        scale detection coordinates and segmentation maps back to the
        original image space after inference.

    Raises:
    -------
    HTTPException : status 406
        If the base64 string is empty or Null
    HTTPException : status 422
        If the base64 string was not able to be processed
    """
    if request.b64_str == "" or request.b64_str is None:
        raise HTTPException(
            status_code=406,
            detail={
                "stage": "image_preprocessing",
                "error": "Base64 string was empty or NULL"
            }
        )
    img = b64_to_image(request.b64_str)

    if img is None:
        raise HTTPException(
            status_code=422,
            detail={
                "stage": "image_preprocessing",
                "error": "Failed to decode base64 string to image."
            }
        )

    orig_size: tuple[int, int] = (img.shape[0], img.shape[1])  # (H, W)

    # Image always gets resized to model input if needed
    if img.shape[0] != MODEL_IMAGE_SIZE or img.shape[1] != MODEL_IMAGE_SIZE:
        img = cv2.resize(img, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    try:
        img = normalize_image(img, means=MEANS, std=STD)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={"stage": "image_preprocessing", "error": str(e)}
        )
    return img[np.newaxis, :, :, :], orig_size  # [1, C, H, W], (H, W)


def order_points(
        pts: np.ndarray, 
        output_size: int = 1024
    ) -> np.ndarray:
    """
    Match 4 source (x, y) points to the TL/TR/BR/BL corners of a square canvas.

    Uses linear sum assignment on Euclidean distance to the four canvas corners
    so that each source point is uniquely assigned to its nearest corner.  This
    is more robust than the sum/diff heuristic when recovered points are mixed
    with detected ones and the quadrilateral is not axis-aligned.

    Parameters:
    -----------
    pts : np.ndarray
        A (4, 2) float32 source points in image space.
    output_size : int
        The side length of the destination canvas (default 1024).

    Returns:
    --------
    ordered : np.ndarray
        A (4, 2) float32 array ordered [TL, TR, BR, BL], ready for
        cv2.getPerspectiveTransform.
    """
    dst_corners = np.array(
        [
            [0, 0],
            [output_size-1, 0],
            [output_size-1, output_size-1],
            [0, output_size-1],
        ], 
        dtype=np.float32
    )
    # Match rows (pts) to columns (dst_corners)
    cost = cdist(pts.reshape(-1, 2), dst_corners)
    row_idx, col_idx = linear_sum_assignment(cost)

    ordered = np.zeros((4, 2), dtype=np.float32)
    for src_i, dst_j in zip(row_idx, col_idx):
        ordered[dst_j] = pts[src_i]
    return ordered


def warp_to_roi(
        class_map: np.ndarray, 
        src_pts: np.ndarray, 
        output_size: int = 1024
    ) -> np.ndarray:
    """
    Perspective-warp a class map so that src_pts map to the four corners.

    Parameters:
    -----------
    class_map : np.ndarray
        HW uint8 array of class indices.
    src_pts : np.ndarray
        (4, 2) float32 array of (x, y) midpoints in image space.
    output_size : int
        Integer side length of the square output canvas (default 1024).

    Returns:
    --------
    warped : np.ndarray
        Warped class map of shape (output_size, output_size) uint8.
    """
    ordered = order_points(src_pts, output_size)
    dst = np.array(
        [
            [0, 0],
            [output_size-1, 0],
            [output_size-1, output_size-1],
            [0, output_size-1],
        ], 
        dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(ordered, dst)

    warped = cv2.warpPerspective(
        class_map.astype(np.uint8),
        M,
        (output_size, output_size),
        flags=cv2.INTER_NEAREST
    )

    return warped


def compute_class_proportions(
        class_map: np.ndarray, 
        class_names: list
    ) -> dict:
    """
    Return the proportion of pixels belonging to each class in class_map.

    Parameters:
    -----------
    class_map : np.ndarray
        Array of class indices (0-indexed) of shape H,W and type uint8.
    class_names : list
        An ordered list of class name strings.

    Returns:
    --------
        Dict mapping class_name -> proportion (0.0–1.0), rounded to 4 decimal places.
    """
    total = class_map.size
    if total == 0:
        return {
            name: 0.0 
            for name in class_names
        }
    
    return {
        name: round(float(np.sum(class_map == idx)) / total, 4)
        for idx, name in enumerate(class_names)
    }


def _points_inside_any_bbox(
        points: np.ndarray, 
        bboxes: np.ndarray
    ) -> np.ndarray:
    """
    Return a boolean mask: True where a point falls inside any (x1, y1, x2, y2) bbox.
    """
    if len(points) == 0 or bboxes is None or len(bboxes) == 0:
        return np.zeros(len(points), dtype=bool)
    x, y = points[:, 0:1], points[:, 1:2]           # [P, 1]
    print("x:", x, "y:", y)
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]  # [B]
    inside = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)  # [P, B]
    print(inside)

    return inside.any(axis=1)


def _rects_covered_by_bboxes(
        rects: np.ndarray, 
        bboxes: np.ndarray, 
        overlap_thresh: float = 0.6
    ) -> np.ndarray:
    """
    Return a boolean mask: True where a contour's own bounding rect is mostly
    (>= overlap_thresh of its area) contained within one of the detected boxes.

    This distinguishes a contour that's just a second blob of an *already-detected*
    marker/corner (its rect sits almost entirely inside that detection's box) from a
    separate, undetected one that merely grazes the edge of a nearby detection's box
    (low overlap, so it must stay eligible for recovery). A plain "is the centroid
    inside any box" check would incorrectly exclude the latter.

    Parameters:
    -----------
    rects : np.ndarray
        (P, 4) float32 array of (x1, y1, x2, y2) contour bounding rects.
    bboxes : np.ndarray
        (K, 4) float32 array of (x1, y1, x2, y2) detected boxes.
    overlap_thresh : float
        Minimum overlap ratio to consider a rect as covered (default 0.6).
    """
    if len(rects) == 0 or bboxes is None or len(bboxes) == 0:
        return np.zeros(len(rects), dtype=bool)

    rx1, ry1, rx2, ry2 = rects[:, 0:1], rects[:, 1:2], rects[:, 2:3], rects[:, 3:4]  # [P, 1]
    bx1, by1, bx2, by2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]      # [B]

    ix1 = np.maximum(rx1, bx1)
    iy1 = np.maximum(ry1, by1)
    ix2 = np.minimum(rx2, bx2)
    iy2 = np.minimum(ry2, by2)

    # Calculate the intersection of each rect with each box and each rect's area
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)  # [P, B]
    rect_area = np.clip((rx2 - rx1) * (ry2 - ry1), 1e-6, None)  # [P, 1]
    overlap_ratio = inter / rect_area  # [P, B]

    return (overlap_ratio >= overlap_thresh).any(axis=1)


def find_marker_midpoints_from_mask(
    class_map: np.ndarray,
    marker_cls_idx: int,
    detected_midpoints: np.ndarray,
    n_missing: int,
    min_contour_area: int = 300,
    detected_bboxes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Recover up to n_missing marker positions from the segmentation class map.

    Algorithm:
    ----------
      1. Threshold the marker class to get a binary mask.
      2. Find external contours; compute each centroid via image moments.
      3. Discard any contour whose own bounding rect is mostly contained within
         an already-detected bounding box (handles a single detection whose mask
         splits into multiple contours, e.g. a marker segmented as two blobs,
         without discarding a separate, genuinely undetected marker that merely
         grazes the edge of a nearby detection's box).
      4. If detections exist, use linear sum assignment to match them to the
         nearest remaining contours, then exclude those matched contours too.
      5. From what's left, greedily select the n_missing centroids that
         maximise the minimum distance to all already-selected points
         (detected + previously recovered).

    Parameters:
    -----------
    class_map : np.ndarray
        HW uint8 segformer output map.
    marker_cls_idx : int
        Class index in class_map that represents markers.
        Pass -1 to disable (returns empty array).
    detected_midpoints : np.ndarray
        (K, 2) float32 array of already-detected (x, y) centers.
    n_missing : int
        Number of marker positions still needed.
    min_contour_area : int
        Discard contours smaller than this (pixels).
    detected_bboxes : np.ndarray | None
        (K, 4) float32 array of already-detected (x1, y1, x2, y2)
        boxes, used to discard contours already covered by a
        real detection. Optional.

    Returns:
    --------
    np.ndarray : 
        (M, 2) float32 array of recovered (x, y) positions, M <= n_missing.
    """
    if marker_cls_idx < 0 or n_missing <= 0:
        return np.empty((0, 2), dtype=np.float32)

    marker_mask = (class_map == marker_cls_idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute (area, centroid, bounding rect) for valid contours
    valid: list[tuple[float, np.ndarray, np.ndarray]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_contour_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        # Compute the centroid coordinates
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        # Grab the contours bbox starting corner and dimensions
        rx, ry, rw, rh = cv2.boundingRect(c)
        rect = np.array([rx, ry, rx + rw, ry + rh], dtype=np.float32)
        valid.append(
            (area, np.array([cx, cy], dtype=np.float32), rect)
        )

    if not valid:
        return np.empty((0, 2), dtype=np.float32)

    # Sort largest-first so the greedy pass prefers prominent markers
    valid.sort(key=lambda x: x[0], reverse=True)
    centroids = np.array([v[1] for v in valid], dtype=np.float32)  # [M, 2]
    rects = np.array([v[2] for v in valid], dtype=np.float32)      # [M, 4]

    # Drop contours already covered by a real detection's bounding box. This is the
    # sole exclusion mechanism - a forced nearest-neighbor match against `detected`
    # would pair every detected point with *something*, even a genuinely separate,
    # undetected marker, once its own duplicate blob has already been dropped here.
    covered = _rects_covered_by_bboxes(rects, detected_bboxes)
    centroids = centroids[~covered]
    if len(centroids) == 0:
        return np.empty((0, 2), dtype=np.float32)

    detected = np.asarray(detected_midpoints, dtype=np.float32).reshape(-1, 2)
    candidates = list(centroids)
    if not candidates:
        return np.empty((0, 2), dtype=np.float32)

    # Greedy max-min-distance selection to favour well-spread-out points
    anchor_pts = list(detected) if len(detected) > 0 else []
    recovered: list[np.ndarray] = []

    for _ in range(min(n_missing, len(candidates))):
        if not anchor_pts:
            chosen = candidates.pop(0)
        else:
            anchor_arr = np.array(anchor_pts, dtype=np.float32)
            min_dists = np.array([
                np.min(np.linalg.norm(anchor_arr - pt, axis=1))
                for pt in candidates
            ])
            best = int(np.argmax(min_dists))
            chosen = candidates.pop(best)
        recovered.append(chosen)
        anchor_pts.append(chosen)

    return np.array(recovered, dtype=np.float32) if recovered else np.empty((0, 2), dtype=np.float32)


def find_quadrat_corners_from_mask(
    class_map: np.ndarray,
    quadrat_cls_idx: int,
    n_missing: int,
    min_contour_area: int = 500,
    detected_bboxes: np.ndarray | None = None,
) -> np.ndarray:
    """Recover up to n_missing quadrat corner positions from the segmentation mask.

    Algorithm:
      1. Threshold the quadrat class to get a binary mask.
      2. Dilate to reconnect any fragmented frame segments.
      3. Take the largest contour and compute its convex hull.
      4. Approximate the hull to 4 corners via cv2.approxPolyDP (falling back
         to cv2.minAreaRect if no 4-vertex approximation is found).
      5. Discard any extracted corner that falls inside an already-detected
         bounding box, and return up to n_missing of what's left.

    Args:
        class_map:          HW uint8 segformer output map.
        quadrat_cls_idx:    Class index for the quadrat frame/corner pixels.
                            Pass -1 to disable (returns empty array).
        n_missing:          Number of corner positions still needed.
        min_contour_area:   Discard contours smaller than this (pixels).
        detected_bboxes:    (K, 4) float32 array of already-detected (x1, y1, x2, y2)
                            boxes, used to discard corners already covered by a
                            real detection. Optional.

    Returns:
        (M, 2) float32 array of recovered corner positions, M <= n_missing.
    """
    if quadrat_cls_idx < 0 or n_missing <= 0:
        return np.empty((0, 2), dtype=np.float32)

    quadrat_mask = (class_map == quadrat_cls_idx).astype(np.uint8) * 255

    # Dilate to bridge gaps between PVC pipe segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(quadrat_mask, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_contour_area:
        return np.empty((0, 2), dtype=np.float32)

    # Convex hull → approximate to 4 corners
    hull = cv2.convexHull(largest)
    peri = cv2.arcLength(hull, True)
    corners: np.ndarray | None = None
    for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            break

    if corners is None:
        # Fallback: minimum-area enclosing rectangle always gives 4 corners
        rect = cv2.minAreaRect(largest)
        corners = cv2.boxPoints(rect).astype(np.float32)

    # Drop corners already covered by a real detection's bounding box.
    covered = _points_inside_any_bbox(corners, detected_bboxes)
    corners = corners[~covered]
    if len(corners) == 0:
        return np.empty((0, 2), dtype=np.float32)

    result = np.array(corners[:n_missing], dtype=np.float32)

    return result if len(result) > 0 else np.empty((0, 2), dtype=np.float32)

def _scale_dets(dets: np.ndarray, orig_size: tuple[int, int]) -> np.ndarray:
    """Scale detection coords from MODEL_IMAGE_SIZE space to orig_size (H, W)."""
    orig_h, orig_w = orig_size
    if orig_h == MODEL_IMAGE_SIZE and orig_w == MODEL_IMAGE_SIZE:
        return dets
    dets = dets.copy()
    sx = orig_w / MODEL_IMAGE_SIZE
    sy = orig_h / MODEL_IMAGE_SIZE
    dets[:, [0, 2]] *= sx  # x1, x2
    dets[:, [1, 3]] *= sy  # y1, y2
    return dets


if __name__ == "__main__":
    """Quick manual test for utils._points_inside_any_bbox."""


    # 3 points, 1 box: point-count != box-count
    points = np.array(
        [
            [1, 1],       # inside the box
            [9, 9],       # outside the box
            [100, 100],   # far outside
            [3, 3],
            [8, 8],
            # [200, 200]
        ], 
        dtype=np.float32
    )
    bboxes = np.array(
        [
            [0, 0, 5, 5],
            [10, 20, 40, 30],
            [17, 100, 30, 120]
        ], 
        dtype=np.float32
    )

    result = _points_inside_any_bbox(points, bboxes)
    print(result)  # expect [ True False False]

    # 2 points, 3 boxes: each point inside a different box
    # points2 = np.array([
    #     [2, 2],    # inside box[0]
    #     [12, 12],  # inside box[1]
    # ], dtype=np.float32)
    # bboxes2 = np.array([
    #     [0, 0, 5, 5],
    #     [10, 10, 15, 15],
    #     [20, 20, 25, 25],
    # ], dtype=np.float32)

    # result2 = _points_inside_any_bbox(points2, bboxes2)
    # print("2 points vs 3 boxes:", result2)  # expect [ True  True]

    # # Edge cases
    # print("empty points:", _points_inside_any_bbox(np.empty((0, 2)), bboxes))
    # print("empty boxes: ", _points_inside_any_bbox(points, np.empty((0, 4))))
    # print("None boxes:  ", _points_inside_any_bbox(points, None))
