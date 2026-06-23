#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

cd /workspace/fashn-vton-1.5

python -m uvicorn api:app --host 0.0.0.0 --port 8000
