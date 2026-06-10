"""
app.py
Main application file
BoMeyering 2026
"""

import os
from dotenv import load_dotenv
from logging import getLogger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models import InferenceRequest, InferenceResponse, RawInferenceResponse
from src.inference import run_inference, run_raw_inference

load_dotenv()

PORT = os.getenv("PORT", 80)
print(PORT)

# Create FastAPI app
app = FastAPI()

# Set up logger
logger = getLogger()

# Global variable to track requests
request_count = 0

# Health check endpoint; required for Runpod to monitor worker health
@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

# Standard inference endpoint. Returns output map and bboxes with standard parameters.
@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    inference_dict = run_inference(request.b64_str)

    return inference_dict

# Raw inference endpoint. Returns the raw model logits and bbox arrays for the end user.
@app.post("/raw_infer", response_model=RawInferenceResponse)
async def raw_infer(request: InferenceRequest):
    raw_inference_dict = run_raw_inference(request.b64_str)

    return raw_inference_dict

# A simple endpoint to show request stats
@app.get("/stats")
async def stats():
    return {"total_requests": request_count}

# Run the app when the script is executed
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 80))
    logger.info(f"Starting PGCView v2 server on port {port}")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)