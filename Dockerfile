FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
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

