# 12.8.x is the first CUDA toolkit with native Blackwell (sm_100 / sm_120)
# codegen. cudnn-runtime (not -base) ships the full CUDA runtime + cuDNN 9:
# libcudart / libcublas / libcublasLt / libcudnn / libcufft / libcurand / libnvrtc.
# onnxruntime-gpu's CUDAExecutionProvider needs all of these at load time or it
# silently falls back to CPUExecutionProvider.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision from the cu128 index: their bundled CUDA 12.8 + cuDNN 9.8
# include Blackwell (sm_120) kernels. Pinned explicitly so pip can't resolve the
# cu126 wheel from PyPI instead.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 torchvision==0.24.1

# Everything else from PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY class_mapping.json .

COPY onnx/ ./onnx/

COPY src/ ./src/

COPY .env .env

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Start the handler
CMD ["./entrypoint.sh"]
