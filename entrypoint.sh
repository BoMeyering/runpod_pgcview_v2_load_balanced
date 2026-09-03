#!/bin/bash
set -e

# The cudnn-runtime base image already puts the full CUDA runtime + cuDNN on the
# default loader path, so onnxruntime-gpu's CUDAExecutionProvider can find
# libcudart / libcublas(Lt) / libcudnn / libcufft / libcurand / libnvrtc.
#
# torch also ships its own copies of these as nvidia-*-cu12 wheels under
# site-packages/nvidia/*/lib. Add every one of those dirs to LD_LIBRARY_PATH too
# so the resolver has a complete, self-consistent set regardless of which the
# CUDA provider binds to first.
NVIDIA_LIBS="$(python3 - <<'PY'
import glob, os
try:
    import nvidia
    base = os.path.dirname(nvidia.__file__)
    print(":".join(sorted(glob.glob(os.path.join(base, "*", "lib")))))
except Exception:
    print("")
PY
)"
export LD_LIBRARY_PATH="${NVIDIA_LIBS:+${NVIDIA_LIBS}:}${LD_LIBRARY_PATH}"

exec python3 app.py
