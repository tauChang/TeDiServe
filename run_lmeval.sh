#!/bin/bash

# ----------------------
BASE_DIR=./experiment_dir
DATE_DIR=$(date +"%Y%m%d")
TIME_DIR=$(date +"%H%M%S")
EXPERIMENT_DIR="${BASE_DIR}/${DATE_DIR}/${TIME_DIR}"
mkdir -p $EXPERIMENT_DIR

ln -sfn "$EXPERIMENT_DIR" ./current_experiment
# copy this file to experiment dir for record keeping
cp run_lmeval.sh $EXPERIMENT_DIR/

SLO=3.2  # seconds
# VLLM Args
MODEL=GSAI-ML/LLaDA-8B-Instruct
NUM_GPUS_PER_MODEL_EXECUTOR=1
# NUM_GPUS_PER_MODEL_EXECUTOR=1,1,1,1
# SCHEDULER_CLASS=vllm.v1.core.sched.cluster_scheduler.ClusterScheduler
SCHEDULER_CLASS=vllm.v1.core.sched.infaas_scheduler.InFaaSScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.llumnix_scheduler.LlumnixScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.tedi_scheduler.TeDiScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.tedi_light_scheduler.TeDiLightScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.urgent_opportunistic_scheduler_max_best_effort.UrgentOpportunisticScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.urgent_opportunistic_scheduler_min_best_effort.UrgentOpportunisticScheduler
# SCHEDULER_CLASS=vllm.v1.core.sched.urgent_opportunistic_with_budget_scheduler.UrgentOpportunisticWithBudgetScheduler
DEFAULT_CONFIDENCE_THRESHOLD=0.9
NUM_PROFILE_RUNS=8
NUM_PROFILE_WARMUP_RUNS=3
STEP_ESTIMATOR_MODEL_CLASS=vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine.LightGradientBoostMachine
# KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
# CONF=599
# STEP_DATA_DIR=./step_data_dynamic_${CONF}
STEP_DATA_DIR=$EXPERIMENT_DIR
LOG_DIR=$EXPERIMENT_DIR/logs
mkdir -p $LOG_DIR

DENOISE_BLOCK_SIZE=32
CACHE_PREFIX=false
CACHE_SUFFIX=false
# CACHE_PREFIX=true
# CACHE_SUFFIX=true

if [ "$CACHE_PREFIX" = true ] && [ "$CACHE_SUFFIX" = true ]; then
    # use dual_cache_256_32 model
    # STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/dual_cache_256_32/model.bin
    # STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/dual_cache_256_32/features.txt
    # STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/dual_cache_32_1120/model.bin
    # STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/dual_cache_32_1120/features.txt
    # STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/dual_cache_1120_256_32/model.bin
    # STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/dual_cache_1120_256_32/features.txt
    # STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/dual_cache_1120_256_32_tiny/model.bin
    # STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/dual_cache_1120_256_32_tiny/features.txt
    STEP_ESTIMATOR_MODEL_PATH=/u/tchang85/dllm/analysis/denoise_step_prediction/models/lgb/1121_dual_256_512_1024_avg/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=/u/tchang85/dllm/analysis/denoise_step_prediction/models/lgb/1121_dual_256_512_1024_avg/features.txt
    # STEP_ESTIMATOR_MODEL_PATH=/u/tchang85/dllm/analysis/denoise_step_prediction/models/lgb/1121_dual_256_avg/model.bin
    # STEP_ESTIMATOR_FEATURES_PATH=/u/tchang85/dllm/analysis/denoise_step_prediction/models/lgb/1121_dual_256_avg/features.txt
    # STEP_ESTIMATOR_MODEL_PATH=/u/tchang85/dllm/analysis/denoise_step_prediction/models/lgb/1121_dual_256_512_1024_quantile_0.7/model.bin
    # STEP_ESTIMATOR_FEATURES_PATH=/u/tchang85/dllm/analysis/denoise_step_prediction/models/lgb/1121_dual_256_512_1024_quantile_0.7/features.txt
elif [ "$CACHE_PREFIX" = true ] && [ "$CACHE_SUFFIX" = false ]; then
    STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/prefix_cache_256_32/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/prefix_cache_256_32/features.txt
elif [ "$CACHE_PREFIX" = false ] && [ "$CACHE_SUFFIX" = false ]; then
    STEP_ESTIMATOR_MODEL_PATH=./analysis/denoise_step_prediction/models/lgb/no_cache_256_32/model.bin
    STEP_ESTIMATOR_FEATURES_PATH=./analysis/denoise_step_prediction/models/lgb/no_cache_256_32/features.txt
fi

# ----------------------
STEP_DATA_FILE=${STEP_DATA_DIR}/step_data.json
WORKLOAD_FILE=${EXPERIMENT_DIR}/workload_history.json
REQUEST_PLOTS_DIR=${EXPERIMENT_DIR}/request_plots

# ----------------------
# LMEval Args
TASK=gsm8k
# LIMIT=100
# AVG_INTER_ARRIVAL_TIME=0.5
# ARRIVAL_PATTERN="50:1:0,150:0.5:0"
# ARRIVAL_PATTERN="300:0.5,300:0.2"
# ARRIVAL_PATTERN="30:3:8"
# ARRIVAL_PATTERN="100:0.8:1"
ARRIVAL_PATTERN="200:1.25"
# ARRIVAL_PATTERN="100:0.1"
# ARRIVAL_PATTERN="1319:0.3125"
# ARRIVAL_PATTERN="5:0"
# ARRIVAL_PATTERN="100:1:0"
# Calculate TOTAL_NUM_REQUESTS based on ARRIVAL_PATTERN
if [ -n "$ARRIVAL_PATTERN" ]; then
    TOTAL_NUM_REQUESTS=$(echo "$ARRIVAL_PATTERN" | awk -F, '{sum=0; for (i=1; i<=NF; i++) {split($i, a, ":"); sum+=a[1]} print sum}')
else
    TOTAL_NUM_REQUESTS=$LIMIT
fi
NUM_CONCURRENT=$TOTAL_NUM_REQUESTS
# NUM_CONCURRENT=1
OUTPUT_LENGTH=256
WRITE_RESULTS=true
RESULTS_DIR=$EXPERIMENT_DIR/results
# make results_dir prefix with confidence
# RESULTS_DIR="${RESULTS_DIR}_conf${DEFAULT_CONFIDENCE_THRESHOLD}"

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
    --experiment-dir ${EXPERIMENT_DIR} \
    --total-num-requests ${TOTAL_NUM_REQUESTS} \
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

EVAL_CMD="python eval/run_lmeval.py \
    --model $MODEL \
    --task $TASK \
    --output-length $OUTPUT_LENGTH \
    --output-path $OUTPUT_PATH \
    --num-concurrent $NUM_CONCURRENT \
    --warmup"

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
# tmux send-keys -t $SESSION "$VLLM_CMD 2>&1 | tee $EXPERIMENT_DIR/vllm_serve.log dllm_serve_multi.log" C-m
tmux send-keys -t $SESSION "$VLLM_CMD 2>&1 | tee $LOG_DIR/vllm_serve.log" C-m

# Split into 3 vertical panes
tmux split-window -h -t $SESSION
tmux split-window -h -t $SESSION

# Pane 2: eval script
# tmux send-keys -t $SESSION.1 "$EVAL_CMD 2>&1 | tee $EXPERIMENT_DIR/lmeval_run.log && python analysis/confidence_over_time/plot.py --step-data $STEP_DATA_FILE --workload-history $WORKLOAD_FILE --output-dir $REQUEST_PLOTS_DIR &&
# python analysis/slo_attainment_and_good_accuracy/run.py --path $OUTPUT_PATH --slo $SLO 2>&1 | tee $EXPERIMENT_DIR/result_summary.log" C-m
# tmux send-keys -t $SESSION.1 "$EVAL_CMD 2>&1 | tee \"$LOG_DIR/lmeval_run.log\"" C-m
# tmux send-keys -t $SESSION.1 "python analysis/confidence_over_time/plot.py --step-data \"$STEP_DATA_FILE\" --workload-history \"$WORKLOAD_FILE\" --output-dir \"$REQUEST_PLOTS_DIR\" 2>&1 | tee \"$LOG_DIR/confidence_over_time.log\"" C-m
# tmux send-keys -t $SESSION.1 "python analysis/profiler_analysis/scheduler/run_schedule.py \"$EXPERIMENT_DIR/profiles/scheduler/schedule.jsonl\" 2>&1 | tee \"$LOG_DIR/schedule_analysis.log\"" C-m
# tmux send-keys -t $SESSION.1 "python analysis/profiler_analysis/scheduler/run_update.py \"$EXPERIMENT_DIR/profiles/scheduler/update.jsonl\" 2>&1 | tee \"$LOG_DIR/update_analysis.log\"" C-m
# tmux send-keys -t $SESSION.1 "python analysis/profiler_analysis/model_runner/run.py \"$EXPERIMENT_DIR/profiles/model_runners/\" 2>&1 | tee \"$LOG_DIR/model_runner_analysis.log\"" C-m
# tmux send-keys -t $SESSION.1 "python analysis/profiler_analysis/step_estimator/run.py \"$EXPERIMENT_DIR/profiles/step_estimator/predict.jsonl\" 2>&1 | tee \"$LOG_DIR/step_estimator_analysis.log\"" C-m
# tmux send-keys -t $SESSION.1 "python analysis/slo_attainment_and_good_accuracy/run.py --path \"$OUTPUT_PATH\" --slo \"$SLO\" 2>&1 | tee \"$LOG_DIR/result_summary.log\"" C-m# Clear previous command history for this experiment
log_and_send() {
    local session_pane="$1"
    shift
    local cmd="$@"
    echo "$cmd" >> "$EXPERIMENT_DIR/commands.txt"
    tmux send-keys -t "$session_pane" "$cmd" C-m
}

echo "# Commands used for analysis" > "$EXPERIMENT_DIR/commands.txt"

log_and_send "$SESSION.1" \
    "$EVAL_CMD 2>&1 | tee \"$LOG_DIR/lmeval_run.log\""

log_and_send "$SESSION.1" \
    "python analysis/confidence_over_time/plot.py --step-data \"$STEP_DATA_FILE\" --workload-history \"$WORKLOAD_FILE\" --output-dir \"$REQUEST_PLOTS_DIR\" 2>&1 | tee \"$LOG_DIR/confidence_over_time.log\""

log_and_send "$SESSION.1" \
    "python analysis/profiler_analysis/scheduler/run_schedule.py \"$EXPERIMENT_DIR/profiles/scheduler/schedule.jsonl\" 2>&1 | tee \"$LOG_DIR/schedule_analysis.log\""

log_and_send "$SESSION.1" \
    "python analysis/profiler_analysis/scheduler/run_update.py \"$EXPERIMENT_DIR/profiles/scheduler/update.jsonl\" 2>&1 | tee \"$LOG_DIR/update_analysis.log\""

log_and_send "$SESSION.1" \
    "python analysis/profiler_analysis/model_runner/run.py \"$EXPERIMENT_DIR/profiles/model_runners/\" 2>&1 | tee \"$LOG_DIR/model_runner_analysis.log\""

log_and_send "$SESSION.1" \
    "python analysis/profiler_analysis/step_estimator/run.py \"$EXPERIMENT_DIR/profiles/step_estimator/predict.jsonl\" 2>&1 | tee \"$LOG_DIR/step_estimator_analysis.log\""

log_and_send "$SESSION.1" \
    "python analysis/profiler_analysis/executor/run.py \"$EXPERIMENT_DIR/profiles/executors\" 2>&1 | tee \"$LOG_DIR/executor_analysis.log\""

log_and_send "$SESSION.1" \
    "python analysis/slo_attainment_and_good_accuracy/run.py --path \"$OUTPUT_PATH\" --slo \"$SLO\" 2>&1 | tee \"$LOG_DIR/result_summary.log\""


# Pane 3: nvidia-smi monitor
tmux send-keys -t $SESSION.2 "watch -n 0.1 nvidia-smi" C-m

# Balance layout into 3 equal vertical columns
tmux select-layout -t $SESSION even-horizontal

# Attach so you can see all three side-by-side
tmux attach -t $SESSION