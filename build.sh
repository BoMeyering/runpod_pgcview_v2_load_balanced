#!/bin/bash
# Build the image
docker build --platform linux/amd64 -t bmeyering/pgcview-runpod-api:v2.0.0 . 

# docker push YOUR_DOCKER_USERNAME/loadbalancer-example:v1.0