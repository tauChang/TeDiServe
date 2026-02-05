#!/bin/bash
set -e

BASE_DIR=/u/tchang85/dllm/tp_experiment_dir/vllm/
DATE_DIR=$(date +"%Y%m%d")
TIME_DIR=$(date +"%H%M%S")
EXPERIMENT_DIR="${BASE_DIR}/${DATE_DIR}/${TIME_DIR}"
export VLLM_TORCH_PROFILER_DIR=${EXPERIMENT_DIR}/torch_profiler
mkdir -p ${EXPERIMENT_DIR}
mkdir -p ${VLLM_TORCH_PROFILER_DIR}
pip show vllm

MODEL="meta-llama/Meta-Llama-3.1-8B"
PORT=8007

CUDA_VISIBLE_DEVICES=0,1 \
vllm serve $MODEL \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --port $PORT \
    --no-enable-prefix-caching \
    2>&1 | tee ${EXPERIMENT_DIR}/vllm_serve.log