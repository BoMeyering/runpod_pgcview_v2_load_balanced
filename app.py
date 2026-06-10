"""
app.py
Main application file
BoMeyering 2026
"""

import os
import numpy as np
from dotenv import load_dotenv
from logging import getLogger
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from src.models import InferenceRequest, InferenceResponse, RawInferenceResponse
from src.inference import SegformerInference, EfficientDetInference
from src.utils import preprocess

# Get environment variables
load_dotenv()
PORT = os.getenv("PORT", 8080)
SEGFORMER_MODEL_PATH = os.getenv("SEGFORMER_MODEL_PATH", "models/segformer.onnx")
EFFICIENTDET_MODEL_PATH = os.getenv("EFFICIENTDET_MODEL_PATH", "models/efficientdet.onnx")


# Create FastAPI app
app = FastAPI()

# Set up logger
logger = getLogger()

# Global variable to track requests
request_count = 0

# Load models at startup
try:
    segformer_inference = SegformerInference(SEGFORMER_MODEL_PATH)
    efficientdet_inference = EfficientDetInference(EFFICIENTDET_MODEL_PATH)
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise RuntimeError(f"Model loading failed: {e}")

# Health check endpoint; required for Runpod to monitor worker health
@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

# Standard inference endpoint. Returns output map and bboxes with standard parameters.
@app.post("/inference", response_model=InferenceResponse)
async def infer(request: InferenceRequest, conf: float = Query(0.85, ge=0.0, le=1.0)):
    
    img = preprocess(request)
    segmentation_results = segformer_inference.run(img)
    detection_results = efficientdet_inference.run(img)
    
    return InferenceResponse(
        output_map=segmentation_results,
        bboxes=detection_results)

# Raw inference endpoint. Returns the raw model logits and bbox arrays for the end user.
@app.post("/raw_inference", response_model=RawInferenceResponse)
async def raw_infer(request: InferenceRequest, conf: float = Query(0.85, ge=0.0, le=1.0)):
    img = preprocess(request)
    segmentation_results = segformer_inference.run(img)
    detection_results = efficientdet_inference.run(img)
    
    return RawInferenceResponse(
        logits=segmentation_results,
        bboxes=detection_results
    )

# A simple endpoint to show request stats
@app.get("/stats")
async def stats():
    return {"total_requests": request_count}

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 80))
    logger.info(f"Starting PGCView v2 server on port {port}")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)