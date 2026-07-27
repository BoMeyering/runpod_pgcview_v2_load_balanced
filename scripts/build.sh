#!/bin/bash
# Build the image
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=$(cat "${SCRIPT_DIR}/../VERSION")

docker build --platform linux/amd64 -t bmeyering/pgcview-runpod-api:v${VERSION} .
