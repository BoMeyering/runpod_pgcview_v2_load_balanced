import numpy as np
import cv2
import base64

from PIL import Image
from io import BytesIO

img_path = "test_images/0d52da71-4f21-4547-a6a3-6456b553bf37.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
original_size = img.shape[:2] # (height, width)

img = cv2.resize(img, (1024, 1024))

# Convert to base64 using OpenCV
_, buffer = cv2.imencode('.png', img)
b64_str = base64.b64encode(buffer).decode('utf-8')
# bytes_data = base64.b64decode(b64_str)

# img_array = np.frombuffer(bytes_data, dtype=np.uint8)

# cv2_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR_RGB)
# # cv2_img = cv2.resize(cv2_img, (original_size[1], original_size[0]))
# cv2.imwrite("reconstructed_image_cv2.png", cv2_img)

import os
import requests

payload = {
    "b64_str": b64_str,
    "conf": 0.85
}

# response = requests.post("http://localhost:8081/full_pipeline", json=payload)

API_KEY = os.environ["RUNPOD_API_KEY"]
URL = "https://cds0121q75mfue.api.runpod.ai/full_pipeline"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(url=URL, headers=headers, json=payload)
print(response.status_code)
print(response.json())