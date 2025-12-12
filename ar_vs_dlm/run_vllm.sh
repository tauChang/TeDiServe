#!/bin/bash
set -e

MODEL="meta-llama/Llama-3.1-8B"
PORT=8004

# Use all GPUs or specify e.g. 0,1
CUDA_VISIBLE_DEVICES=0 \

vllm serve $MODEL \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --port $PORT
