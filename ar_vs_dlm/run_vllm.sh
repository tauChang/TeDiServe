#!/bin/bash
set -e
pip show vllm

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
PORT=8004

# Use all GPUs or specify e.g. 0,1
CUDA_VISIBLE_DEVICES=0 \
vllm serve $MODEL \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port $PORT \
    --no-enable-prefix-caching \
    --compilation-config {\"full_cuda_graph\":true} \
    --distributed-executor-backend ray \
    2>&1
