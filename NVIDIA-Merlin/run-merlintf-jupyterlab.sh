#!/bin/bash

exec docker run -it --rm --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 \
    -v .://workspace/data/ --ipc=host \
    -w //workspace/data \
    nvcr.io/nvidia/merlin/merlin-tensorflow:nightly \
    jupyter lab --allow-root --ip='0.0.0.0'
