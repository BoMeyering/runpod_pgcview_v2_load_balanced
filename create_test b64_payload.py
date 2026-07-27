"""
Create test base64 payload for the inference endpoint
BoMeyering 2026
Oxbow Solutions, LLC
"""

import cv2
import base64
import numpy as np

def create_b64_img_payload(img_path: str) -> str:
    """
    Create a base64 encoded string from an image file.

    Parameters:
    -----------
    img_path: str
        The file path to the input image.

    Returns:
    --------
    b64_str: str
        A base64 encoded string representing the input image, suitable for use as a payload in the inference API.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))
    flag, buffer = cv2.imencode('.png', img)
    if not flag:
        raise ValueError(f"Failed to read or encode the image at path: {img_path}")
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return b64_str

img_path = "0a5eedf5-6a91-4f72-ab6e-c60989b925e1.jpg"

b64_payload = create_b64_img_payload(img_path)

with open("test_payload.txt", "w") as f:
    f.write(b64_payload)
