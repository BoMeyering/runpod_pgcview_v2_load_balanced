"""
src/models.py
Pydantic request and response models
BoMeyering 2026
"""

from pydantic import BaseModel

# Define request models
class InferenceRequest(BaseModel):
    b64_str: str # a base64 encoded .png image of shape (3, 1024, 1024)

class InferenceResponse(BaseModel):
    output_map: str # A base64 encoded .png binary output map
    bboxes: str # A base64 encoded array of filtered bounding boxes

class RawInferenceResponse(BaseModel):
    logits: str # A base64 encoded array of logits
    bboxes: str # A base64 encoded array of all bounding boxes and classes