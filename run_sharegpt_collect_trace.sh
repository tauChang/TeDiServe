#!/bin/bash

##############################################
# Experiment directory setup (mirrors run_lmeval.sh)
##############################################
VLLM_LOGGING_LEVEL=INFO
BASE_DIR=/work/hdd/bftv/tchang85/experiment_dir
DATE_DIR=$(date +"%Y%m%d")
TIME_DIR=$(date +"%H%M%S")
EXPERIMENT_DIR="${BASE_DIR}/${DATE_DIR}/${TIME_DIR}"
mkdir -p "$EXPERIMENT_DIR"

ln -sfn "$EXPERIMENT_DIR" ./current_experiment
cp run_sharegpt.sh "$EXPERIMENT_DIR/"

LOG_DIR="$EXPERIMENT_DIR/logs"
mkdir -p "$LOG_DIR"

WORKLOAD_FILE=${EXPERIMENT_DIR}/workload_history.json
REQUEST_PLOTS_DIR=${EXPERIMENT_DIR}/request_plots
##############################################
# VLLM / Scheduler / Step Estimator config
# (mirrors structure of run_lmeval.sh)
##############################################

# VLLM Args
MODEL="GSAI-ML/LLaDA-8B-Instruct"
NUM_GPUS_PER_MODEL_EXECUTOR=1
SCHEDULER_CLASS=vllm.v1.core.sched.infaas_scheduler.InFaaSScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.infaas_aligned_scheduler.InFaaSAlignedScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.tedi_new_correct_tput_scheduler.TeDiLightScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.llumnix_fcfs_scheduler.LlumnixFCFSScheduler
DEFAULT_CONFIDENCE_THRESHOLD=0.5
NUM_PROFILE_RUNS=8
NUM_PROFILE_WARMUP_RUNS=3
STEP_ESTIMATOR_MODEL_CLASS=vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine.LightGradientBoostMachine

# KV transfer config (optional)
# KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
KV_TRANSFER_CONFIG=""

STEP_DATA_DIR="$EXPERIMENT_DIR"
STEP_DATA_FILE="${STEP_DATA_DIR}/step_data.json"

DENOISE_BLOCK_SIZE=32
CACHE_PREFIX=true
CACHE_SUFFIX=true

# STEP ESTIMATOR MODEL SELECTION MIRRORED FROM YOUR SCRIPT
if [ "$CACHE_PREFIX" = true ] && [ "$CACHE_SUFFIX" = true ]; then
    # STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/1121_dual_256_512_1024_avg/model.bin
    # STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/1121_dual_256_512_1024_avg/features.txt
    STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/sharegpt_1130/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/sharegpt_1130/features.txt
elif [ "$CACHE_PREFIX" = true ] && [ "$CACHE_SUFFIX" = false ]; then
    STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/prefix_cache_256_32/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/prefix_cache_256_32/features.txt
elif [ "$CACHE_PREFIX" = false ] && [ "$CACHE_SUFFIX" = false ]; then
    STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/no_cache_256_32/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/no_cache_256_32/features.txt
fi

##############################################
# Default ShareGPT benchmark configuration
# (from your run_sharegpt-style script)
##############################################
TOKENIZER="GSAI-ML/LLaDA-8B-Instruct"
IP_PORTS=("localhost:8100")
DATASET_PATH="eval/sharegpt_gpt4_clean_train.jsonl"
MAX_REQUEST_LEN=4096
LOG_LATENCIES=0
FAIL_ON_ERROR=0
VERBOSE=1
TRUST_REMOTE_CODE=0
# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_20qps_trunc.txt"
# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_20qps_75_to_90.txt"
# ARRIVAL_TIME_FILE="/u/tchang85/dllm/BurstGPT/burstgpt_2hrs_20qps_75_to_90.txt"
# ARRIVAL_TIME_FILE="/u/tchang85/dllm/BurstGPT/burstgpt_2hrs_22qps_75_to_90.txt"
# ARRIVAL_TIME_FILE="/u/tchang85/dllm/BurstGPT/burstgpt_2hrs_24qps_75_to_85.txt"
# ARRIVAL_TIME_FILE="/u/tchang85/dllm/BurstGPT/burstgpt_2hrs_23qps_75_to_90.txt"
# ARRIVAL_TIME_FILE="/u/tchang85/dllm/BurstGPT/burstgpt_2hrs_24qps.txt"
# ARRIVAL_TIME_FILE="/u/tchang85/dllm/BurstGPT/burstgpt_2hrs_22qps.txt"
# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_24qps_75_to_85.txt"
# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_20qps.txt"

# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_20qps_20_to_60.txt"
# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_17qps_20_to_60.txt"
# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_18qps_20_to_60.txt"
# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_8hrs_35qps_170_to_190.txt"
# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/arrival_trace_4_12_20.txt"

# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_22qps.txt"
# ARRIVAL_TIME_FILE="/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_24qps.txt"
LIMIT=10
NUM_REQUESTS=100
SLO=12.5

##############################################
# Build VLLM command (mirroring run_lmeval.sh)
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
    # --enforce-eager \

# mirror cache flags
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

##############################################
# Build ShareGPT benchmark command
##############################################
SHAREGPT_CMD="python eval/run_sharegpt.py \
    --tokenizer \"$TOKENIZER\" \
    --max_request_len $MAX_REQUEST_LEN \
    --dataset_path \"$DATASET_PATH\" \
    --arrival_time_file \"$ARRIVAL_TIME_FILE\" \
    --log_filename \"$LOG_DIR/benchmark.log\""

# add limit and num_requests if specified
if [ -n "$LIMIT" ]; then
    SHAREGPT_CMD+=" --limit $LIMIT"
fi
if [ -n "$NUM_REQUESTS" ]; then
    SHAREGPT_CMD+=" --num_requests $NUM_REQUESTS"
fi

# ip_ports
for ip in "${IP_PORTS[@]}"; do
    SHAREGPT_CMD+=" --ip_ports $ip"
done

# options
[[ $LOG_LATENCIES -eq 1 ]] && SHAREGPT_CMD+=" --log_latencies"
[[ $FAIL_ON_ERROR -eq 1 ]] && SHAREGPT_CMD+=" --fail_on_response_failure"
[[ $VERBOSE -eq 1 ]] && SHAREGPT_CMD+=" --verbose"
[[ $TRUST_REMOTE_CODE -eq 1 ]] && SHAREGPT_CMD+=" --trust_remote_code"
[[ -n "$RANDOM_PROMPT_COUNT" ]] && SHAREGPT_CMD+=" --random_prompt_count $RANDOM_PROMPT_COUNT"

##############################################
# tmux + logging (mirrors your run_lmeval.sh)
##############################################
SESSION=sharegpt_session
tmux new-session -d -s $SESSION

log_and_send() {
    local session_pane="$1"
    shift
    local cmd="$@"
    echo "$cmd" >> "$EXPERIMENT_DIR/commands.txt"
    tmux send-keys -t "$session_pane" "$cmd" C-m
}

echo "# Commands used for this experiment" > "$EXPERIMENT_DIR/commands.txt"

# Pane 0: vllm serve
log_and_send "$SESSION.0" \
    "export VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL && $VLLM_CMD 2>&1 | tee \"$LOG_DIR/vllm_serve.log\""

# Pane 1: ShareGPT loadgen
tmux split-window -h -t $SESSION
log_and_send "$SESSION.1" \
    "$SHAREGPT_CMD 2>&1 | tee \"$LOG_DIR/sharegpt_run.log\""

log_and_send "$SESSION.1" "python analysis/sharegpt_slo_attainment/run.py \
    --path \"$LOG_DIR/benchmark_latency_info.json\" \
    --slo $SLO | tee -a \"$LOG_DIR/slo_attainment.log\""

# log_and_send "$SESSION.1" \
#     "python analysis/confidence_over_time/plot.py --step-data \"$STEP_DATA_FILE\" --workload-history \"$WORKLOAD_FILE\" --output-dir \"$REQUEST_PLOTS_DIR\" 2>&1 | tee \"$LOG_DIR/confidence_over_time.log\""

# Pane 2: nvidia-smi monitor
tmux split-window -h -t $SESSION
tmux send-keys -t "$SESSION.2" "watch -n 0.1 nvidia-smi" C-m

tmux select-layout -t $SESSION even-horizontal
tmux attach -t $SESSION
