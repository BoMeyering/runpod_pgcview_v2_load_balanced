#!/bin/bash
set -e

# onnxruntime-gpu's CUDAExecutionProvider needs libcublasLt/libcudnn at
# runtime, but doesn't install them itself. requirements.txt pulls in the
# nvidia-cublas-cu12 / nvidia-cudnn-cu12 pip wheels for these; point the
# loader at their bundled .so files here since pip doesn't do this for us.
CUDNN_LIB="$(python3 -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))')"
CUBLAS_LIB="$(python3 -c 'import nvidia.cublas, os; print(os.path.join(list(nvidia.cublas.__path__)[0], "lib"))')"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${CUBLAS_LIB}:${LD_LIBRARY_PATH}"

exec python3 app.py
