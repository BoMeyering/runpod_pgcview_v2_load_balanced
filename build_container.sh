#!/bin/bash

set -euo pipefail

IMAGE_NAME="ghcr.io/bomeyering/pgcview-runpod-api"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <VERSION>" >&2
    echo "Example: $0 0.0.2" >&2
    exit 1
fi

VERSION="$1"

echo "Building ${IMAGE_NAME}:v${VERSION}"

docker build --platform linux/amd64 -t "${IMAGE_NAME}:v${VERSION}" -t "${IMAGE_NAME}:latest" .