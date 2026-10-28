#!/bin/bash
curl \
-X GET https://mm45wxdx8ku3z1.api.runpod.ai/openapi.json \
-H "Authorization: Bearer ${RUNPOD_API_KEY:?set RUNPOD_API_KEY}" \
-H "Content-Type: application/json" \
-H "Accept: application/json" \