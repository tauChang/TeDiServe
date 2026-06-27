#!/bin/bash
#SBATCH --job-name=dllm-lmeval
#SBATCH --output=sbatch_experiment_dir/slurm-%x-%j.out
#SBATCH --error=sbatch_experiment_dir/slurm-%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=ghx4       # Adjust to your cluster's partition/queue
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --account=bftv-dtai-gh
#SBATCH --gpus=2
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tchang85@wisc.edu

set -euo pipefail

echo "Running on host $(hostname) at path $(pwd)"
BASE_DIR="./experiment_dir"
DATE_DIR=$(date +"%Y%m%d")
TIME_DIR=$(date +"%H%M%S")
EXPERIMENT_DIR="${BASE_DIR}/${DATE_DIR}/${TIME_DIR}"
mkdir -p "$EXPERIMENT_DIR"

ln -sfn "$EXPERIMENT_DIR" ./current_experiment
# copy this file to experiment dir for record keeping
cp run_lmeval.sh $EXPERIMENT_DIR/

LOG_DIR="$EXPERIMENT_DIR/logs"
mkdir -p "$LOG_DIR"

SLO=${SLO:-10}
VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}
NUM_GPUS_PER_MODEL_EXECUTOR=${NUM_GPUS_PER_MODEL_EXECUTOR:-1}
DISTRIBUTED_EXECUTOR_BACKEND=${DISTRIBUTED_EXECUTOR_BACKEND:-ray}
SYNC_STEP_PREDICTION=${SYNC_STEP_PREDICTION:-false}
SCHEDULER_CLASS=${SCHEDULER_CLASS:-vllm.v1.core.sched.tedi_new_correct_tput_async_update_scheduler.TeDiLightScheduler}
DEFAULT_CONFIDENCE_THRESHOLD=${DEFAULT_CONFIDENCE_THRESHOLD:-0.9}
CANDIDATE_CONFIDENCE_THRESHOLDS=${CANDIDATE_CONFIDENCE_THRESHOLDS:-"0.9 0.8 0.7 0.6 0.5"}
STEP_ESTIMATOR_REFRESH_UNMASKED_TOKEN_DELTA=${STEP_ESTIMATOR_REFRESH_UNMASKED_TOKEN_DELTA:-1}
CONFIDENCE_THRESHOLD_TPUT_DEMAND_CHANGE_RATIO=${CONFIDENCE_THRESHOLD_TPUT_DEMAND_CHANGE_RATIO:-100000000}
NUM_PROFILE_RUNS=${NUM_PROFILE_RUNS:-8}
NUM_PROFILE_WARMUP_RUNS=${NUM_PROFILE_WARMUP_RUNS:-3}
STEP_ESTIMATOR_MODEL_CLASS=${STEP_ESTIMATOR_MODEL_CLASS:-vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine.LightGradientBoostMachine}
STEP_DATA_DIR=${STEP_DATA_DIR:-$EXPERIMENT_DIR}
DENOISE_BLOCK_SIZE=${DENOISE_BLOCK_SIZE:-32}
CACHE_PREFIX=${CACHE_PREFIX:-true}
CACHE_SUFFIX=${CACHE_SUFFIX:-true}
TASK=${TASK:-gsm8k}
ARRIVAL_PATTERN=${ARRIVAL_PATTERN:-"20:0"}
OUTPUT_LENGTH=${OUTPUT_LENGTH:-256}
WRITE_RESULTS=${WRITE_RESULTS:-true}
VLLM_PORT=${VLLM_PORT:-8000}
NUM_CONCURRENT=${NUM_CONCURRENT:-}

if [ -z "$NUM_CONCURRENT" ]; then
    if [ -n "$ARRIVAL_PATTERN" ]; then
        TOTAL_NUM_REQUESTS=$(echo "$ARRIVAL_PATTERN" | awk -F, '{sum=0; for (i=1; i<=NF; i++) {split($i, a, ":"); sum+=a[1]} print sum}')
    else
        TOTAL_NUM_REQUESTS=${TOTAL_NUM_REQUESTS:-0}
    fi

    if [ "$TASK" = "mmlu_pro" ]; then
        TOTAL_NUM_REQUESTS=$(( TOTAL_NUM_REQUESTS * 14 ))
    fi

    NUM_CONCURRENT=$TOTAL_NUM_REQUESTS
else
    TOTAL_NUM_REQUESTS=${TOTAL_NUM_REQUESTS:-$NUM_CONCURRENT}
fi

STEP_ESTIMATOR_MODEL_PATH=${STEP_ESTIMATOR_MODEL_PATH:-./analysis/denoise_step_prediction/models/lgb/sharegpt_0415_all_features/model.bin}
STEP_ESTIMATOR_FEATURES_PATH=${STEP_ESTIMATOR_FEATURES_PATH:-./analysis/denoise_step_prediction/models/lgb/sharegpt_0415_all_features/features.txt}
RESULTS_DIR="$EXPERIMENT_DIR/results"
mkdir -p "$RESULTS_DIR"

STEP_DATA_FILE="${STEP_DATA_DIR}/step_data.json"
WORKLOAD_FILE="${EXPERIMENT_DIR}/workload_history.json"
REQUEST_PLOTS_DIR="${EXPERIMENT_DIR}/request_plots"

SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '_')
CACHE_PREFIX_STR=$([ "$CACHE_PREFIX" = true ] && echo "_prefix" || echo "")
CACHE_SUFFIX_STR=$([ "$CACHE_SUFFIX" = true ] && echo "_suffix" || echo "")
if [ "$DENOISE_BLOCK_SIZE" -gt -1 ]; then
    BLOCK_SIZE_STR="_block${DENOISE_BLOCK_SIZE}"
else
    BLOCK_SIZE_STR=""
fi
OUTPUT_PATH="${RESULTS_DIR}/${TASK}_/${OUTPUT_LENGTH}/${SAFE_MODEL_NAME}${CACHE_PREFIX_STR}${CACHE_SUFFIX_STR}${BLOCK_SIZE_STR}.json"

VLLM_CMD="vllm serve --trust-remote-code ${MODEL} \
    --distributed-executor-backend ${DISTRIBUTED_EXECUTOR_BACKEND} \
    --num-gpus-per-model-executor ${NUM_GPUS_PER_MODEL_EXECUTOR} \
    --scheduler_cls ${SCHEDULER_CLASS} \
    --default-confidence-threshold ${DEFAULT_CONFIDENCE_THRESHOLD} \
    --candidate-confidence-thresholds ${CANDIDATE_CONFIDENCE_THRESHOLDS} \
    --step-estimator-refresh-unmasked-token-delta ${STEP_ESTIMATOR_REFRESH_UNMASKED_TOKEN_DELTA} \
    --confidence-threshold-tput-demand-change-ratio ${CONFIDENCE_THRESHOLD_TPUT_DEMAND_CHANGE_RATIO} \
    --num-profile-runs ${NUM_PROFILE_RUNS} \
    --num-profile-warmup-runs ${NUM_PROFILE_WARMUP_RUNS} \
    --step-estimator-model-class ${STEP_ESTIMATOR_MODEL_CLASS} \
    --step-estimator-model-path ${STEP_ESTIMATOR_MODEL_PATH} \
    --step-estimator-features-path ${STEP_ESTIMATOR_FEATURES_PATH} \
    --step-data-dir ${STEP_DATA_DIR} \
    --experiment-dir ${EXPERIMENT_DIR} \
    --total-num-requests ${TOTAL_NUM_REQUESTS} \
    --port ${VLLM_PORT} \
    --compilation-config '{\"full_cuda_graph\": true}'"

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
    --num-concurrent $NUM_CONCURRENT \
    --warmup"

if [ -n "$ARRIVAL_PATTERN" ]; then
    EVAL_CMD="$EVAL_CMD --arrival-pattern \"$ARRIVAL_PATTERN\""
fi
if [ "$WRITE_RESULTS" = "true" ]; then
    EVAL_CMD="$EVAL_CMD --write-results"
fi

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
    local timeout_seconds=1200
    local start_time
    start_time=$(date +%s)

    echo "Waiting for vLLM on port ${VLLM_PORT}..."
    while true; do
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
            return 0
        fi

        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "vLLM server exited before it became ready"
            return 1
        fi

        if (( $(date +%s) - start_time >= timeout_seconds )); then
            echo "Timed out waiting for vLLM"
            return 1
        fi

        sleep 1
    done
}

run_logged() {
    local cmd="$1"
    local logfile="$2"
    echo "$cmd" >> "$EXPERIMENT_DIR/commands.txt"
    bash -lc "$cmd" 2>&1 | tee "$logfile"
}

echo "# Commands used for analysis" > "$EXPERIMENT_DIR/commands.txt"

export VLLM_LOGGING_LEVEL

setsid bash -lc "$VLLM_CMD 2>&1 | tee \"$LOG_DIR/vllm_serve.log\"" &
server_pid=$!

wait_for_server

run_logged "$EVAL_CMD" "$LOG_DIR/lmeval_run.log"
run_logged "python analysis/slo_attainment_and_good_accuracy/run.py --path \"$OUTPUT_PATH\" --slo \"$SLO\"" "$LOG_DIR/result_summary.log"

if [[ "$SCHEDULER_CLASS" == *TeDi* ]]; then
    run_logged "python analysis/confidence_over_time/plot.py --step-data \"$STEP_DATA_FILE\" --workload-history \"$WORKLOAD_FILE\" --output-dir \"$REQUEST_PLOTS_DIR\"" "$LOG_DIR/confidence_over_time.log"
fi

run_logged "python analysis/profiler_analysis/scheduler/run_schedule.py \"$EXPERIMENT_DIR/profiles/scheduler/schedule.jsonl\"" "$LOG_DIR/schedule_analysis.log"
run_logged "python analysis/calculate_num_recompute/run.py --path \"$EXPERIMENT_DIR/system_log.json\"" "$LOG_DIR/num_recompute.log"
run_logged "python analysis/profiler_analysis/scheduler/run_update.py \"$EXPERIMENT_DIR/profiles/scheduler/update.jsonl\"" "$LOG_DIR/update_analysis.log"
run_logged "python analysis/profiler_analysis/model_runner/run.py \"$EXPERIMENT_DIR/profiles/model_runners/\"" "$LOG_DIR/model_runner_analysis.log"
run_logged "python analysis/profiler_analysis/step_estimator/run.py \"$EXPERIMENT_DIR/profiles/step_estimator/predict.jsonl\"" "$LOG_DIR/step_estimator_analysis.log"
run_logged "python analysis/profiler_analysis/executor/run.py \"$EXPERIMENT_DIR/profiles/executors\"" "$LOG_DIR/executor_analysis.log"
run_logged "python analysis/experiment_summary/run.py --result-path \"$OUTPUT_PATH\" --slo \"$SLO\" --scheduler-summary \"$EXPERIMENT_DIR/profiles/scheduler/schedule_summary.txt\" --predict-summary \"$EXPERIMENT_DIR/profiles/step_estimator/predict_summary.txt\" --confidence-stats \"$REQUEST_PLOTS_DIR/avg_confidence_stats.json\"" "$LOG_DIR/experiment_summary.log"