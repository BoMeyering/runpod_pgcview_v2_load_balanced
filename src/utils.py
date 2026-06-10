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
from io import BytesIO
from logging import getLogger
from fastapi import HTTPException

logger = getLogger()

# Load environment variables for means and stds; provide defaults if not set
load_dotenv()
MEANS = np.array(os.getenv("MEANS", "0.485,0.456,0.406").split(","), dtype=np.float32)
STD = np.array(os.getenv("STDS", "0.229,0.224,0.225").split(","), dtype=np.float32)

def b64_to_image(b64_str: str) -> np.ndarray:
    """
    Convert a base64 encoded string to a numpy array image.

    Parameters:
    -----------
    b64_str: str
        A base64 encoded string representing an image.
        This should be in HWC format, in RGB order, a .png binary.

    Returns:
    --------
    img: np.ndarray
        A numpy array representing the decoded image, in HWC format, RGB order.

    Raises:
    -------
    HTTPException:
        If there is an error during base64 decoding or image decoding, an HTTPException is raised with details about the failure.
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

def normalize_image(img: np.ndarray, means: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image for model inference.

    Parameters:
    -----------
    img: np.ndarray
        A numpy array representing the input image, in HWC format, RGB order.

    Returns:
    --------
    preprocessed_img: np.ndarray
        A numpy array representing the preprocessed image, ready for model input.
        This should be in CHW format, in RGB order, and normalized.
    """

    img = img.astype(np.float32) / 255.0
    img = (img - means.astype(np.float32)) / std.astype(np.float32)
  
    preprocessed_img = np.transpose(img, (2, 0, 1))

    return preprocessed_img

def grayscale_image_to_b64(img: np.ndarray) -> str:
    """
    Converst a grayscale image to a base64 encoded string
    
    Parameters:
    -----------
    img: np.ndarray
        A numpy array representing a grayscale image in HW format, with pixel values in [0, 255]
    
    Returns:
    --------
    b64_str: str
        A base64 encoded string representing the input grayscale image as a .png binary
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
    b64_str = base64.b64encode(buffer).decode('utf-8')

    return b64_str

def array_to_b64(arr: np.ndarray) -> str:
    """
    Convert a numpy array to a base64 encoded string.

    Parameters:
    -----------
    arr: np.ndarray
        A numpy array to be converted to a base64 string.

    Returns:
    --------
    b64_str: str
        A base64 encoded string representing the input array.
    """

    bytes_data = arr.tobytes()
    b64_str = base64.b64encode(bytes_data).decode('utf-8')

    return b64_str

def preprocess(request):
    """
    Preprocess the input image from a base64 encoded string.
    
    Parameters:
    -----------
    request: InferenceRequest
        The incoming request containing the base64 encoded image string.

    Returns:
    --------
    img: np.ndarray
        A numpy array representing the preprocessed image, ready for model input.

    Raises:
    -------
    HTTPException:
        If there is an error during image decoding or preprocessing, an HTTPException is raised with details about the failure.
    """
    try:
        b64_str = request.b64_str
        img = b64_to_image(b64_str)
        if img is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "stage": "image_preprocessing",
                    "error": "Failed to decode base64 string to image. Ensure the input is a valid base64 encoded .png image in RGB format."
                }
            )
        img = normalize_image(img, means=MEANS, std=STD)
    except Exception as e:
        raise HTTPException(
            status_code=422, 
            detail={
                "stage": "image_preprocessing",
                "error": str(e)
            }
        )
    
    return img