#!/bin/bash

# docker run -it --platform linux/amd64 --gpus all -p 8080:8080 bmeyering/pgcview-runpod-api:v2.0.0
    # -v $(pwd)/data:/app/data \
    # -v $(pwd)/models:/app/models \
    # -v $(pwd)/logs:/app/logs \
    
docker run -d --name pgcview_v2 --platform linux/amd64 -p 8081:80 bmeyering/pgcview-runpod-api:v2.0.0