#!/bin/bash
set -e

MODEL="meta-llama/Meta-Llama-3.1-8B"
PORT=8007

CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve $MODEL \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --port $PORT
