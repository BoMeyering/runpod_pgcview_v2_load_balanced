# This RunPod endpoint's GPU pool is Ampere/Ada only (A4000/A4500/A5000/A6000/A40,
# RTX 2000/4000/4090, L4, 3090, MIG 24GB slices) - no Blackwell (sm_100/120) cards.
# 12.6 covers all of those and only requires driver >=560 (vs >=570 for 12.8),
# which is what actually broke deploys onto older-driver hosts. Bump back to
# 12.8+ only if Blackwell GPUs are ever added to this endpoint's pool.
# cudnn-runtime (not -base) ships the full CUDA runtime + cuDNN 9:
# libcudart / libcublas / libcublasLt / libcudnn / libcufft / libcurand / libnvrtc.
# onnxruntime-gpu's CUDAExecutionProvider needs all of these at load time or it
# silently falls back to CPUExecutionProvider.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.description="PGCView V2 inference API for RunPod serverless. Runs Segformer segmentation and EfficientDet marker/quadrat detection on field images to measure per-class vegetation cover and active/dormant grass fractions within the detected ROI."

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision from the cu126 index (the oldest CUDA index PyPI publishes
# torch==2.9.1 for). Pinned explicitly so pip can't resolve the cu128 wheel
# instead.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.9.1 torchvision==0.24.1

# Everything else from PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY class_mapping.json .

COPY onnx/ ./onnx/

COPY src/ ./src/

# Non-secret runtime config only; the local .env holds API keys and must not be baked in.
COPY docker.env .env

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Start the handler
CMD ["./entrypoint.sh"]
