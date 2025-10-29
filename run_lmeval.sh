#!/bin/bash

# ----------------------
# VLLM Args
MODEL=GSAI-ML/LLaDA-8B-Instruct
NUM_GPUS_PER_MODEL_EXECUTOR=4,2,1,1
# SCHEDULER_CLASS=vllm.v1.core.sched.cluster_scheduler.ClusterScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.dynamic_confidence_scheduler.ClusterScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.urgent_scheduler.UrgentOpportunisticScheduler
SCHEDULER_CLASS=vllm.v1.core.sched.urgent_opportunistic_scheduler.UrgentOpportunisticScheduler
DEFAULT_CONFIDENCE_THRESHOLD=0.9
NUM_PROFILE_RUNS=8
NUM_PROFILE_WARMUP_RUNS=3
STEP_ESTIMATOR_MODEL_CLASS=vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine.LightGradientBoostMachine
STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/model.bin
STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/features.txt
# CONF=599
# STEP_DATA_DIR=./step_data_dynamic_${CONF}
STEP_DATA_DIR=./step_data_1024_default_1.5

DENOISE_BLOCK_SIZE=32
CACHE_PREFIX=false
CACHE_SUFFIX=false

# ----------------------
# LMEval Args
TASK=gsm8k
# LIMIT=100
# AVG_INTER_ARRIVAL_TIME=0.5
# ARRIVAL_PATTERN="100:3,100:1,100:2"
ARRIVAL_PATTERN="200:1.5"
NUM_CONCURRENT=200
OUTPUT_LENGTH=256
WRITE_RESULTS=true
RESULTS_DIR=eval/results_1024_default_1.5

# ----------------------
# Output path name
SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '_')
CACHE_PREFIX_STR=$([ "$CACHE_PREFIX" = true ] && echo "_prefix" || echo "")
CACHE_SUFFIX_STR=$([ "$CACHE_SUFFIX" = true ] && echo "_suffix" || echo "")
if [ "$DENOISE_BLOCK_SIZE" -gt -1 ]; then
    BLOCK_SIZE_STR="_block${DENOISE_BLOCK_SIZE}"
else
    BLOCK_SIZE_STR=""
fi
CONFIDENCE_STR="_conf${DEFAULT_CONFIDENCE_THRESHOLD}"
# OUTPUT_PATH="eval/results/${TASK}_${LIMIT}/${OUTPUT_LENGTH}/${SAFE_MODEL_NAME}${CACHE_PREFIX_STR}${CACHE_SUFFIX_STR}${BLOCK_SIZE_STR}${CONFIDENCE_STR}.json"
OUTPUT_PATH="${RESULTS_DIR}/${TASK}_${LIMIT}/${OUTPUT_LENGTH}/${SAFE_MODEL_NAME}${CACHE_PREFIX_STR}${CACHE_SUFFIX_STR}${BLOCK_SIZE_STR}.json"

# ----------------------
# Build the commands
VLLM_CMD="vllm serve --trust-remote-code ${MODEL} \
    --distributed-executor-backend ray \
    --enforce-eager \
    --num-gpus-per-model-executor ${NUM_GPUS_PER_MODEL_EXECUTOR} \
    --scheduler_cls ${SCHEDULER_CLASS} \
    --default-confidence-threshold ${DEFAULT_CONFIDENCE_THRESHOLD} \
    --num-profile-runs ${NUM_PROFILE_RUNS} \
    --num-profile-warmup-runs ${NUM_PROFILE_WARMUP_RUNS} \
    --eval-task ${TASK}_${LIMIT} \
    --gen-len ${OUTPUT_LENGTH} \
    --step-estimator-model-class ${STEP_ESTIMATOR_MODEL_CLASS} \
    --step-estimator-model-path ${STEP_ESTIMATOR_MODEL_PATH} \
    --step-estimator-features-path ${STEP_ESTIMATOR_FEATURES_PATH} \
    --step-data-dir ${STEP_DATA_DIR} \
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

EVAL_CMD="python eval/run_lmeval.py \
    --model $MODEL \
    --task $TASK \
    --output-length $OUTPUT_LENGTH \
    --output-path $OUTPUT_PATH \
    --num-concurrent $NUM_CONCURRENT"

# If ARRIVAL_PATTERN is set → use it and skip limit/avg-inter-arrival
if [ -n "$ARRIVAL_PATTERN" ]; then
    EVAL_CMD="$EVAL_CMD --arrival-pattern \"$ARRIVAL_PATTERN\""
else
    # Otherwise, fall back to simple arrival config
    EVAL_CMD="$EVAL_CMD \
        --limit $LIMIT \
        --avg-inter-arrival-time $AVG_INTER_ARRIVAL_TIME"
fi

# Append optional flags
if [ "$WRITE_RESULTS" = "true" ]; then
    EVAL_CMD="$EVAL_CMD --write-results"
fi

# ----------------------
# Start tmux session with two panes
SESSION=eval_session
tmux new-session -d -s $SESSION

# Pane 1: vllm serve
tmux send-keys -t $SESSION "$VLLM_CMD 2>&1 | tee dllm_serve_multi.log" C-m

# Split into 3 vertical panes
tmux split-window -h -t $SESSION
tmux split-window -h -t $SESSION

# Pane 2: eval script
tmux send-keys -t $SESSION.1 "$EVAL_CMD 2>&1 | tee eval.log" C-m

# Pane 3: nvidia-smi monitor
tmux send-keys -t $SESSION.2 "watch -n 0.5 nvidia-smi" C-m

# Balance layout into 3 equal vertical columns
tmux select-layout -t $SESSION even-horizontal

# Attach so you can see all three side-by-side
tmux attach -t $SESSION