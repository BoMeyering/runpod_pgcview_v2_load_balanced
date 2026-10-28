"""
src/veg_indices.py
RGB vegetation index helpers
BoMeyering 2024
"""

import cv2
import numpy as np


def clahe_channel(img: np.ndarray, clip_limit=100) -> np.ndarray:
    """
    Performs CLAHE on the l* channel of an image

    Args:
        img (np.ndarray): a three channel image in opencv format (BGR order)

    Returns:
        np.ndarray: The adjusted image
    """
    # Convert to lab
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Apply CLAHE
    l_channel = lab_img[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    new_l = clahe.apply(l_channel)
    lab_img[:, :, 0] = new_l

    # Convert back to BGR
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)

    return bgr_img
