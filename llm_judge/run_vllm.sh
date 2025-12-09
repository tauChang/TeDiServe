#!/bin/bash
set -e

MODEL="meta-llama/Llama-3.3-70B-Instruct"
PORT=8007

# Use all GPUs or specify e.g. 0,1
CUDA_VISIBLE_DEVICES=0,1,2,3 \

vllm serve $MODEL \
    --tensor-parallel-size 4 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --port $PORT
