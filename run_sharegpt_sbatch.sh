#!/bin/bash

set -euo pipefail

##############################################
# Experiment directory setup
##############################################
BASE_DIR=./experiment_dir
DATE_DIR=$(date +"%Y%m%d")
TIME_DIR=$(date +"%H%M%S")
EXPERIMENT_DIR="${BASE_DIR}/${DATE_DIR}/${TIME_DIR}"
mkdir -p "$EXPERIMENT_DIR"

ln -sfn "$EXPERIMENT_DIR" ./current_experiment
cp run_sharegpt_sbatch.sh "$EXPERIMENT_DIR/"

LOG_DIR="$EXPERIMENT_DIR/logs"
mkdir -p "$LOG_DIR"

WORKLOAD_FILE=${EXPERIMENT_DIR}/workload_history.json
REQUEST_PLOTS_DIR=${EXPERIMENT_DIR}/request_plots

##############################################
# VLLM / Scheduler / Step Estimator config
##############################################
VLLM_LOGGING_LEVEL=DEBUG
MODEL="GSAI-ML/LLaDA-8B-Instruct"
NUM_GPUS_PER_MODEL_EXECUTOR=1
SCHEDULER_CLASS=vllm.v1.core.sched.tedi_new_correct_tput_async_update_scheduler.TeDiLightScheduler
DEFAULT_CONFIDENCE_THRESHOLD=0.9
NUM_PROFILE_RUNS=8
NUM_PROFILE_WARMUP_RUNS=3
STEP_ESTIMATOR_MODEL_CLASS=vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine.LightGradientBoostMachine

KV_TRANSFER_CONFIG=""

STEP_DATA_DIR="$EXPERIMENT_DIR"
STEP_DATA_FILE="${STEP_DATA_DIR}/step_data.json"

DENOISE_BLOCK_SIZE=32
CACHE_PREFIX=true
CACHE_SUFFIX=true

if [ "$CACHE_PREFIX" = true ] && [ "$CACHE_SUFFIX" = true ]; then
    STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/sharegpt_0415_all_features/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/sharegpt_0415_all_features/features.txt
elif [ "$CACHE_PREFIX" = true ] && [ "$CACHE_SUFFIX" = false ]; then
    STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/prefix_cache_256_32/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/prefix_cache_256_32/features.txt
elif [ "$CACHE_PREFIX" = false ] && [ "$CACHE_SUFFIX" = false ]; then
    STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/no_cache_256_32/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/no_cache_256_32/features.txt
fi

##############################################
# Default ShareGPT benchmark configuration
##############################################
TOKENIZER="GSAI-ML/LLaDA-8B-Instruct"
IP_PORTS=("localhost:8080")
DATASET_PATH="eval/sharegpt_gpt4_clean_train.jsonl"
MAX_REQUEST_LEN=4096
LOG_LATENCIES=0
FAIL_ON_ERROR=0
VERBOSE=1
TRUST_REMOTE_CODE=0
LIMIT=10
NUM_REQUESTS=10
SLO=12.5

if [ -z "${ARRIVAL_TIME_FILE:-}" ]; then
    echo "ARRIVAL_TIME_FILE must be set before running this script."
    exit 1
fi

##############################################
# Build commands
##############################################
VLLM_CMD="vllm serve --trust-remote-code ${MODEL} \
    --distributed-executor-backend ray \
    --num-gpus-per-model-executor ${NUM_GPUS_PER_MODEL_EXECUTOR} \
    --scheduler_cls ${SCHEDULER_CLASS} \
    --default-confidence-threshold ${DEFAULT_CONFIDENCE_THRESHOLD} \
    --num-profile-runs ${NUM_PROFILE_RUNS} \
    --num-profile-warmup-runs ${NUM_PROFILE_WARMUP_RUNS} \
    --step-estimator-model-class ${STEP_ESTIMATOR_MODEL_CLASS} \
    --step-estimator-model-path ${STEP_ESTIMATOR_MODEL_PATH} \
    --step-estimator-features-path ${STEP_ESTIMATOR_FEATURES_PATH} \
    --step-data-dir ${STEP_DATA_DIR} \
    --experiment-dir ${EXPERIMENT_DIR} \
    --compilation-config '{\"full_cuda_graph\": true}' \
    --port ${IP_PORTS[0]##*:} \
    --request-latency-slo $SLO \
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

SHAREGPT_CMD="python eval/run_sharegpt.py \
    --tokenizer \"$TOKENIZER\" \
    --max_request_len $MAX_REQUEST_LEN \
    --dataset_path \"$DATASET_PATH\" \
    --arrival_time_file \"$ARRIVAL_TIME_FILE\" \
    --log_filename \"$LOG_DIR/benchmark.log\""

if [ -n "${LIMIT:-}" ]; then
    SHAREGPT_CMD+=" --limit $LIMIT"
fi
if [ -n "${NUM_REQUESTS:-}" ]; then
    SHAREGPT_CMD+=" --num_requests $NUM_REQUESTS"
fi

for ip in "${IP_PORTS[@]}"; do
    SHAREGPT_CMD+=" --ip_ports $ip"
done

[[ $LOG_LATENCIES -eq 1 ]] && SHAREGPT_CMD+=" --log_latencies"
[[ $FAIL_ON_ERROR -eq 1 ]] && SHAREGPT_CMD+=" --fail_on_response_failure"
[[ $VERBOSE -eq 1 ]] && SHAREGPT_CMD+=" --verbose"
[[ $TRUST_REMOTE_CODE -eq 1 ]] && SHAREGPT_CMD+=" --trust_remote_code"
[[ -n "${RANDOM_PROMPT_COUNT:-}" ]] && SHAREGPT_CMD+=" --random_prompt_count $RANDOM_PROMPT_COUNT"

##############################################
# Cleanup helpers
##############################################
server_pid=""

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill -- -"$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi

    exit "$exit_code"
}

trap cleanup EXIT INT TERM

wait_for_server() {
    local port=$1
    local timeout_seconds=1200
    local start_time
    start_time=$(date +%s)

    echo "Waiting for server on port $port..."

    while true; do
        if curl -s "localhost:${port}/v1/completions" > /dev/null; then
            return 0
        fi

        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server exited before it became ready"
            return 1
        fi

        local now
        now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then
            echo "Timeout waiting for server"
            return 1
        fi

        sleep 1
    done
}

##############################################
# Run server and benchmark
##############################################
echo "# Commands used for this experiment" > "$EXPERIMENT_DIR/commands.txt"
echo "$VLLM_CMD" >> "$EXPERIMENT_DIR/commands.txt"
echo "$SHAREGPT_CMD" >> "$EXPERIMENT_DIR/commands.txt"

export VLLM_LOGGING_LEVEL

setsid bash -lc "$VLLM_CMD 2>&1 | tee \"$LOG_DIR/vllm_serve.log\"" &
server_pid=$!

wait_for_server "${IP_PORTS[0]##*:}"

bash -lc "$SHAREGPT_CMD 2>&1 | tee \"$LOG_DIR/sharegpt_run.log\""

python analysis/sharegpt_slo_attainment/run.py \
    --path "$LOG_DIR/benchmark_latency_info.json" \
    --slo $SLO | tee -a "$LOG_DIR/slo_attainment.log"
