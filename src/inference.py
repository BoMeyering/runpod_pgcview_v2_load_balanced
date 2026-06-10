"""
inference.py
Inference logic for the PGCView V2 model inference API
BoMeyering 2026
Oxbow Solutions, LLC
"""

from onnxruntime import InferenceSession
import numpy as np
from abc import ABC, abstractmethod
from fastapi import HTTPException

from .utils import b64_to_image

class ModelInference(ABC):
    @abstractmethod
    def __init__(self, model_path):
        self.session = InferenceSession(
            model_path,
            execution_providers=["CPUExecutionProvider"]
        )

    @abstractmethod
    def run(self, img):
        ...


class SegformerInference(ModelInference):
    def __init__(self, model_path):
        super().__init__(model_path)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Implement preprocessing logic specific to the Segformer model
        pass
    
    def run(self, img: np.ndarray):
        """ Run inference on the preprocessed image and return the raw model outputs. """

        try:
            input_name = self.session.get_inputs()[0].name
            inputs = {input_name: img}
            logits = self.session.run(None, inputs)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Segformer model inference failed: {e}"
            )
        
        return logits
    
class EfficientDetInference(ModelInference):
    def __init__(self, model_path):
        super().__init__(model_path)
    
    def run(self, img: np.ndarray):
        """ Run inference on the preprocessed image and return the raw model outputs. """

        try:
            input_name = self.session.get_inputs()[0].name
            inputs = {input_name: img}
            bboxes = self.session.run(None, inputs)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"EfficientDet model inference failed: {e}"
            )
        
        return bboxes