#!/bin/bash

# ----------------------
# BASE_DIR=./experiment_dir
# DATE_DIR=$(date +"%Y%m%d")
# TIME_DIR=$(date +"%H%M%S")
EXPERIMENT_DIR="tmp/"
mkdir -p $EXPERIMENT_DIR

# ln -sfn "$EXPERIMENT_DIR" ./current_experiment
# # copy this file to experiment dir for record keeping
# cp run_lmeval.sh $EXPERIMENT_DIR/

SLO=7.6 # seconds
# VLLM Args
MODEL=GSAI-ML/LLaDA-8B-Instruct
# MODEL=GSAI-ML/LLaDA-8B-Base
NUM_GPUS_PER_MODEL_EXECUTOR=4,

SCHEDULER_CLASS=vllm.v1.core.sched.infaas_aligned_scheduler.InFaaSAlignedScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.llumnix_fcfs_scheduler.LlumnixFCFSScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.tedi_new_correct_tput_scheduler.TeDiLightScheduler

DEFAULT_CONFIDENCE_THRESHOLD=1.01
STEP_ESTIMATOR_MODEL_CLASS=vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine.LightGradientBoostMachine

DENOISE_BLOCK_SIZE=32
# CACHE_PREFIX=false
# CACHE_SUFFIX=false
CACHE_PREFIX=true
CACHE_SUFFIX=true

if [ "$CACHE_PREFIX" = true ] && [ "$CACHE_SUFFIX" = true ]; then
    STEP_ESTIMATOR_MODEL_PATH=../analysis/denoise_step_prediction/models/lgb/1121_dual_256_512_1024_avg/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=../analysis/denoise_step_prediction/models/lgb/1121_dual_256_512_1024_avg/features.txt
elif [ "$CACHE_PREFIX" = true ] && [ "$CACHE_SUFFIX" = false ]; then
    STEP_ESTIMATOR_MODEL_PATH=../analysis/denoise_step_prediction/models/lgb/prefix_cache_256_32/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=../analysis/denoise_step_prediction/models/lgb/prefix_cache_256_32/features.txt
elif [ "$CACHE_PREFIX" = false ] && [ "$CACHE_SUFFIX" = false ]; then
    STEP_ESTIMATOR_MODEL_PATH=../analysis/denoise_step_prediction/models/lgb/no_cache_256_32/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=../analysis/denoise_step_prediction/models/lgb/no_cache_256_32/features.txt
fi

# ----------------------
# Build the commands
VLLM_CMD="vllm serve --trust-remote-code ${MODEL} \
    --distributed-executor-backend ray \
    --enforce-eager \
    --num-gpus-per-model-executor ${NUM_GPUS_PER_MODEL_EXECUTOR} \
    --scheduler_cls ${SCHEDULER_CLASS} \
    --default-confidence-threshold ${DEFAULT_CONFIDENCE_THRESHOLD} \
    --num-profile-runs 1 \
    --num-profile-warmup-runs 1 \
    --eval-task ${TASK}_${LIMIT} \
    --gen-len 1 \
    --step-estimator-model-class ${STEP_ESTIMATOR_MODEL_CLASS} \
    --step-estimator-model-path ${STEP_ESTIMATOR_MODEL_PATH} \
    --step-estimator-features-path ${STEP_ESTIMATOR_FEATURES_PATH} \
    --step-data-dir ${EXPERIMENT_DIR} \
    --experiment-dir ${EXPERIMENT_DIR} \
    --total-num-requests 1 \
    "

if [ "$CACHE_PREFIX" = "true" ]; then
  VLLM_CMD="$VLLM_CMD --cache-prefix"
fi
if [ "$CACHE_SUFFIX" = "true" ]; then
  VLLM_CMD="$VLLM_CMD --cache-suffix"
fi
if [ "$DENOISE_BLOCK_SIZE" -gt -1 ]; then
    VLLM_CMD="$VLLM_CMD --denoise-block-size ${DENOISE_BLOCK_SIZE}"
fi
if [ -n "$KV_TRANSFER_CONFIG" ]; then
    VLLM_CMD="$VLLM_CMD --kv-transfer-config '${KV_TRANSFER_CONFIG}'"
fi

# ----------------------
# Start tmux session with two panes
# SESSION=eval_session
# tmux new-session -d -s $SESSION

# Pane 1: vllm serve
# tmux send-keys -t $SESSION "$VLLM_CMD 2>&1 | tee $EXPERIMENT_DIR/vllm_serve.log dllm_serve_multi.log" C-m
# tmux send-keys -t $SESSION "$VLLM_CMD 2>&1 | tee $LOG_DIR/vllm_serve.log" C-m
$VLLM_CMD 2>&1 | tee dllm_serve.log
