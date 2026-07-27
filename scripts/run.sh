#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=$(cat "${SCRIPT_DIR}/../VERSION")

docker run -d --name pgcview_v2 --platform linux/amd64 -p 3465:80 bmeyering/pgcview-runpod-api:v${VERSION}
