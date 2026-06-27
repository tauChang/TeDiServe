#!/bin/bash
#SBATCH --job-name=dllm-lmeval-multi
#SBATCH --output=sbatch_experiment_dir/slurm-%x-%j.out
#SBATCH --error=sbatch_experiment_dir/slurm-%x-%j.err
#SBATCH --time=01:20:00
#SBATCH --partition=ghx4
#SBATCH --account=bftv-dtai-gh
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=36
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tchang85@wisc.edu

set -euo pipefail

STORAGE_DIR=/work/hdd/bftv/tchang85
SCRIPT_DIR=./
cd "$SCRIPT_DIR"

# CONFIG_FILE=${CONFIG_FILE:-${EXPERIMENTS_CONFIG_FILE:-}}
CONFIG_FILE=${CONFIG_FILE:-}
# make sure config file is not empty
if [ -z "$CONFIG_FILE" ]; then
    echo "CONFIG_FILE or EXPERIMENTS_CONFIG_FILE environment variable must be set to the path of the experiment configuration JSON file."
    exit 1
fi
# CONFIG_FILE=./sbatch_lmeval_config.json
# CONFIG_FILE=./sbatch_config/get_ground_truth.json
# CONFIG_FILE=./sbatch_config/correct_demand.json
# CONFIG_FILE=./sbatch_config/gsm8k_ablation_4workers.json
# CONFIG_FILE=./sbatch_config/tp4_profile.json

# CONFIG_FILE=./sbatch_config/fake_executor_smoke_test.json
# CONFIG_FILE=./sbatch_config/oracle_vs_async8_vs_oneshot_rps.json
# CONFIG_FILE=./sbatch_config/one_shot_with_gsm8k_average.json
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

CONFIG_BASENAME=$(basename "$CONFIG_FILE")
SBATCH_EXPERIMENT_NAME=${SBATCH_EXPERIMENT_NAME:-${CONFIG_BASENAME%.*}}
# Keep sbatch directory names safe and predictable.
SBATCH_EXPERIMENT_NAME=${SBATCH_EXPERIMENT_NAME//[^A-Za-z0-9._-]/_}

DATE_DIR=$(date +"%Y%m%d")
TIME_DIR=$(date +"%H%M%S")
SBATCH_DIR="$STORAGE_DIR/sbatch_experiment_dir/$DATE_DIR/${TIME_DIR}_${SBATCH_EXPERIMENT_NAME}"
mkdir -p "$SBATCH_DIR"

exec > >(tee -a "$SBATCH_DIR/stdout.log") 2> >(tee -a "$SBATCH_DIR/stderr.log" >&2)

EXPERIMENT_RUN_DIRS=()
CURRENT_SERVER_PID=""
RAY_BOOTSTRAP_PID=""
RAY_PORT_SEED=${SLURM_JOB_ID:-$$}
if [ -z "${RAY_PORT:-}" ]; then
    RAY_PORT=$((20000 + (RAY_PORT_SEED % 30) * 1000))
fi
RAY_OBJECT_MANAGER_PORT=${RAY_OBJECT_MANAGER_PORT:-$((RAY_PORT + 1))}
RAY_NODE_MANAGER_PORT=${RAY_NODE_MANAGER_PORT:-$((RAY_PORT + 2))}
RAY_RUNTIME_ENV_AGENT_PORT=${RAY_RUNTIME_ENV_AGENT_PORT:-$((RAY_PORT + 3))}
RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-$((RAY_PORT + 4))}
RAY_DASHBOARD_AGENT_GRPC_PORT=${RAY_DASHBOARD_AGENT_GRPC_PORT:-$((RAY_PORT + 5))}
RAY_METRICS_EXPORT_PORT=${RAY_METRICS_EXPORT_PORT:-$((RAY_PORT + 6))}
RAY_CLIENT_SERVER_PORT=${RAY_CLIENT_SERVER_PORT:-$((RAY_PORT + 7))}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-$((RAY_PORT + 100))}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-$((RAY_PORT + 199))}
RAY_RESET_STATE=${RAY_RESET_STATE:-false}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}
RAY_BOOTSTRAP_LOG="$SBATCH_DIR/ray_bootstrap.log"
RAY_BOOTSTRAP_SCRIPT="$SBATCH_DIR/ray_bootstrap.sh"
export RAY_PORT RAY_OBJECT_MANAGER_PORT RAY_NODE_MANAGER_PORT
export RAY_RUNTIME_ENV_AGENT_PORT RAY_DASHBOARD_AGENT_LISTEN_PORT
export RAY_DASHBOARD_AGENT_GRPC_PORT RAY_METRICS_EXPORT_PORT
export RAY_CLIENT_SERVER_PORT RAY_MIN_WORKER_PORT RAY_MAX_WORKER_PORT
export RAY_RESET_STATE RAY_NUM_CPUS

ln -sfn "$SBATCH_DIR" "$SCRIPT_DIR/current_experiment"
cp "$SCRIPT_DIR/run_lmeval_multi_sbatch.sh" "$SBATCH_DIR/"
cp "$CONFIG_FILE" "$SBATCH_DIR/experiments_config.json"

load_config_exports() {
    local config_file="$1"
    local experiment_index="$2"

    python - "$config_file" "$experiment_index" <<'PY'
import json
import shlex
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
experiment_index = int(sys.argv[2])

config = json.loads(config_path.read_text())
common = config.get("common", {})
experiments = config.get("experiments", [])

if experiment_index < 0 or experiment_index >= len(experiments):
    raise SystemExit(f"experiment index out of range: {experiment_index}")

merged = dict(common)
merged.update(experiments[experiment_index])

name = merged.pop("name", None) or merged.pop("experiment_name", None)
if name is not None:
    merged["EXPERIMENT_NAME"] = name

for key, value in merged.items():
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif value is None:
        value = ""
    elif isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    else:
        value = str(value)
    print(f"{key}={shlex.quote(value)}")
PY
}

load_experiment_names() {
    local config_file="$1"

    python - "$config_file" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
experiments = config.get("experiments", [])

for index, experiment in enumerate(experiments):
    name = experiment.get("name") or experiment.get("experiment_name") or f"experiment_{index + 1}"
    print(name)
PY
}

run_logged() {
    local cmd="$1"
    local logfile="$2"
    echo "$cmd" >> "$EXPERIMENT_DIR/commands.txt"
    bash --noprofile --norc -c "$cmd" 2>&1 | tee "$logfile"
}

run_logged_allow_failure() {
    local cmd="$1"
    local logfile="$2"
    local label="$3"

    echo "$cmd" >> "$EXPERIMENT_DIR/commands.txt"
    if ! bash --noprofile --norc -c "$cmd" 2>&1 | tee "$logfile"; then
        echo "Warning: ${label} failed, continuing with the rest of the batch" >&2
    fi
}

wait_for_server() {
    local port="$1"
    local server_pid="$2"
    local timeout_seconds=1200
    local start_time
    start_time=$(date +%s)

    echo "Waiting for vLLM on port ${port}..."
    while true; do
        if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
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

stop_server() {
    local server_pid="$1"
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill -- -"$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}

stop_ray_cluster() {
    local num_nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"

    if [ -n "${RAY_BOOTSTRAP_PID:-}" ] && kill -0 "$RAY_BOOTSTRAP_PID" 2>/dev/null; then
        kill "$RAY_BOOTSTRAP_PID" 2>/dev/null || true
        wait "$RAY_BOOTSTRAP_PID" 2>/dev/null || true
    fi
}

dump_ray_bootstrap_log() {
    if [ -f "$RAY_BOOTSTRAP_LOG" ]; then
        echo "=== Ray bootstrap log: $RAY_BOOTSTRAP_LOG ==="
        tail -n 200 "$RAY_BOOTSTRAP_LOG" || true
    fi
}

wait_for_ray_cluster() {
    local head_address="$1"
    local timeout_seconds=300
    local start_time
    start_time=$(date +%s)

    echo "Waiting for Ray at ${head_address}:${RAY_PORT}..."
    while true; do
        if ray status --address="${head_address}:${RAY_PORT}" >/dev/null 2>&1; then
            return 0
        fi

        if [ -n "${RAY_BOOTSTRAP_PID:-}" ] && ! kill -0 "$RAY_BOOTSTRAP_PID" 2>/dev/null; then
            echo "Ray bootstrap exited before the cluster became ready"
            dump_ray_bootstrap_log
            return 1
        fi

        if (( $(date +%s) - start_time >= timeout_seconds )); then
            echo "Timed out waiting for Ray cluster"
            dump_ray_bootstrap_log
            return 1
        fi

        sleep 2
    done
}

start_ray_cluster() {
    local num_nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
    if [ "$num_nodes" -le 1 ]; then
        echo "Single-node allocation; skipping external Ray bootstrap"
        return 0
    fi

    if [ -z "${SLURM_NODELIST:-}" ]; then
        echo "SLURM_NODELIST is required for multi-node Ray bootstrap"
        return 1
    fi

    local nodes
    mapfile -t nodes < <(scontrol show hostnames "$SLURM_NODELIST")
    if [ ${#nodes[@]} -eq 0 ]; then
        echo "Unable to resolve nodes from SLURM_NODELIST: $SLURM_NODELIST"
        return 1
    fi

    local head_node="${nodes[0]}"
    local head_address
    head_address=$(getent hosts "$head_node" | awk '{print $1; exit}')
    if [ -z "$head_address" ]; then
        head_address="$head_node"
    fi

    cat > "$RAY_BOOTSTRAP_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail

node_name="$(hostname -s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ray bootstrap on ${node_name}: head=${RAY_HEAD_NODE} address=${RAY_HEAD_ADDRESS} port=${RAY_PORT} cpus=${RAY_NUM_CPUS}"

if [[ "$RAY_RESET_STATE" == "true" ]]; then
    ray stop --force >/dev/null 2>&1 || true
fi

if [[ "$node_name" == "$RAY_HEAD_NODE" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting Ray head on ${node_name}"
    ray start --head \
        --port="$RAY_PORT" \
        --num-cpus="$RAY_NUM_CPUS" \
        --include-dashboard=false \
        --node-manager-port="$RAY_NODE_MANAGER_PORT" \
        --object-manager-port="$RAY_OBJECT_MANAGER_PORT" \
        --runtime-env-agent-port="$RAY_RUNTIME_ENV_AGENT_PORT" \
        --dashboard-agent-listen-port="$RAY_DASHBOARD_AGENT_LISTEN_PORT" \
        --dashboard-agent-grpc-port="$RAY_DASHBOARD_AGENT_GRPC_PORT" \
        --metrics-export-port="$RAY_METRICS_EXPORT_PORT" \
        --ray-client-server-port="$RAY_CLIENT_SERVER_PORT" \
        --min-worker-port="$RAY_MIN_WORKER_PORT" \
        --max-worker-port="$RAY_MAX_WORKER_PORT" \
        --block
else
    sleep 15
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting Ray worker on ${node_name}, connecting to ${RAY_HEAD_ADDRESS}:${RAY_PORT}"
    ray start \
        --address="${RAY_HEAD_ADDRESS}:${RAY_PORT}" \
        --num-cpus="$RAY_NUM_CPUS" \
        --node-manager-port="$RAY_NODE_MANAGER_PORT" \
        --object-manager-port="$RAY_OBJECT_MANAGER_PORT" \
        --runtime-env-agent-port="$RAY_RUNTIME_ENV_AGENT_PORT" \
        --dashboard-agent-listen-port="$RAY_DASHBOARD_AGENT_LISTEN_PORT" \
        --dashboard-agent-grpc-port="$RAY_DASHBOARD_AGENT_GRPC_PORT" \
        --metrics-export-port="$RAY_METRICS_EXPORT_PORT" \
        --min-worker-port="$RAY_MIN_WORKER_PORT" \
        --max-worker-port="$RAY_MAX_WORKER_PORT" \
        --block
fi
EOF
    chmod +x "$RAY_BOOTSTRAP_SCRIPT"

    export RAY_HEAD_NODE="$head_node"
    export RAY_HEAD_ADDRESS="$head_address"
    export RAY_ADDRESS="${head_address}:${RAY_PORT}"

    echo "Starting Ray cluster on ${#nodes[@]} nodes; head node ${head_node} (${head_address}); port=${RAY_PORT}; cpus=${RAY_NUM_CPUS}"
    echo "Ray bootstrap log: $RAY_BOOTSTRAP_LOG"
    srun --nodes="$num_nodes" --ntasks="$num_nodes" --ntasks-per-node=1 --kill-on-bad-exit=1 "$RAY_BOOTSTRAP_SCRIPT" >> "$RAY_BOOTSTRAP_LOG" 2>&1 &
    RAY_BOOTSTRAP_PID=$!

    wait_for_ray_cluster "$head_address"
    echo "Ray cluster is ready"
}

cleanup() {
    if [ -n "${CURRENT_SERVER_PID:-}" ]; then
        stop_server "$CURRENT_SERVER_PID"
    fi
    stop_ray_cluster
}

trap cleanup EXIT

run_one_experiment() {
    local experiment_name="$1"
    local experiment_index="$2"

    export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
    export MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}
    export NUM_GPUS_PER_MODEL_EXECUTOR=${NUM_GPUS_PER_MODEL_EXECUTOR:-1}
    export DISTRIBUTED_EXECUTOR_BACKEND=${DISTRIBUTED_EXECUTOR_BACKEND:-ray}
    export SYNC_STEP_PREDICTION=${SYNC_STEP_PREDICTION:-false}
    export SCHEDULER_CLASS=${SCHEDULER_CLASS:-vllm.v1.core.sched.tedi_new_correct_tput_async_update_scheduler.TeDiLightScheduler}
    export DEFAULT_CONFIDENCE_THRESHOLD=${DEFAULT_CONFIDENCE_THRESHOLD:-0.9}
    export CANDIDATE_CONFIDENCE_THRESHOLDS=${CANDIDATE_CONFIDENCE_THRESHOLDS:-"0.9 0.8 0.7 0.6 0.5"}
    export STEP_ESTIMATOR_REFRESH_UNMASKED_TOKEN_DELTA=${STEP_ESTIMATOR_REFRESH_UNMASKED_TOKEN_DELTA:-1}
    export CONFIDENCE_THRESHOLD_TPUT_DEMAND_CHANGE_RATIO=${CONFIDENCE_THRESHOLD_TPUT_DEMAND_CHANGE_RATIO:-100000000}
    export RECONFIG_INTERVAL=${RECONFIG_INTERVAL:--1}
    export NUM_PROFILE_RUNS=${NUM_PROFILE_RUNS:-8}
    export NUM_PROFILE_WARMUP_RUNS=${NUM_PROFILE_WARMUP_RUNS:-3}
    export STEP_ESTIMATOR_MODEL_CLASS=${STEP_ESTIMATOR_MODEL_CLASS:-vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine.LightGradientBoostMachine}
    export TASK=${TASK:-gsm8k}
    export ARRIVAL_PATTERN=${ARRIVAL_PATTERN:-"20:0"}
    export OUTPUT_LENGTH=${OUTPUT_LENGTH:-256}
    export NUM_FEWSHOT=${NUM_FEWSHOT:-}
    export WRITE_RESULTS=${WRITE_RESULTS:-true}
    export ENABLE_DROPPING=${ENABLE_DROPPING:-false}
    export VLLM_PORT=${VLLM_PORT:-8000}
    export DENOISE_BLOCK_SIZE=${DENOISE_BLOCK_SIZE:-32}
    export CACHE_PREFIX=${CACHE_PREFIX:-true}
    export CACHE_SUFFIX=${CACHE_SUFFIX:-true}
    export STEP_ESTIMATE_UPDATE_INTERVAL=${STEP_ESTIMATE_UPDATE_INTERVAL:-1}
    export MAX_CONFIDENCE_UPDATE_INTERVAL=${MAX_CONFIDENCE_UPDATE_INTERVAL:-1}
    export MAX_NUM_UNFINISHED_REQUESTS=${MAX_NUM_UNFINISHED_REQUESTS:-}
    export NUM_CONCURRENT=${NUM_CONCURRENT:-}

    local run_date_dir
    local run_time_dir
    run_date_dir=$(date +"%Y%m%d")
    run_time_dir=$(date +"%H%M%S")
    # local run_dir="/work/hdd/bftv/tchang85/experiment_dir/$run_date_dir/${run_time_dir}_${experiment_index}_${experiment_name}"
    local run_dir="$SBATCH_DIR/${experiment_index}_${experiment_name}"
    local log_dir="$run_dir/logs"
    local results_dir="$run_dir/results"
    local request_plots_dir="$run_dir/request_plots"

    mkdir -p "$run_dir" "$log_dir" "$results_dir"
    EXPERIMENT_RUN_DIRS+=("$run_dir")
    # ln -sfn "$run_dir" "$SCRIPT_DIR/current_experiment"

    export EXPERIMENT_DIR="$run_dir"
    export STEP_DATA_DIR="$run_dir"
    export WORKLOAD_FILE="$run_dir/workload_history.json"
    export STEP_DATA_FILE="$run_dir/step_data.json"
    export REQUEST_PLOTS_DIR="$request_plots_dir"

    local safe_model_name
    safe_model_name=$(echo "$MODEL" | tr '/' '_')
    local cache_prefix_str=""
    local cache_suffix_str=""
    local block_size_str=""
    if [ "$CACHE_PREFIX" = true ]; then
        cache_prefix_str="_prefix"
    fi
    if [ "$CACHE_SUFFIX" = true ]; then
        cache_suffix_str="_suffix"
    fi
    if [ "$DENOISE_BLOCK_SIZE" -gt -1 ]; then
        block_size_str="_block${DENOISE_BLOCK_SIZE}"
    fi

    local total_num_requests
    if [ -z "${NUM_CONCURRENT}" ]; then
        if [ -n "$ARRIVAL_PATTERN" ]; then
            total_num_requests=$(echo "$ARRIVAL_PATTERN" | awk -F, '{sum=0; for (i=1; i<=NF; i++) {split($i, a, ":"); sum+=a[1]} print sum}')
        else
            total_num_requests=0
        fi

        if [ "$TASK" = "mmlu_pro" ]; then
            total_num_requests=$(( total_num_requests * 14 ))
        fi

        export NUM_CONCURRENT="$total_num_requests"
    else
        total_num_requests="$NUM_CONCURRENT"
    fi

    local output_path="$results_dir/${TASK}_/${OUTPUT_LENGTH}/${safe_model_name}${cache_prefix_str}${cache_suffix_str}${block_size_str}.json"
    mkdir -p "$(dirname "$output_path")"

    local vllm_cmd
    vllm_cmd="vllm serve --trust-remote-code ${MODEL} \
    --distributed-executor-backend ${DISTRIBUTED_EXECUTOR_BACKEND} \
    --num-gpus-per-model-executor ${NUM_GPUS_PER_MODEL_EXECUTOR} \
    --scheduler_cls ${SCHEDULER_CLASS} \
    --request-latency-slo ${SLO} \
    --default-confidence-threshold ${DEFAULT_CONFIDENCE_THRESHOLD} \
    --candidate-confidence-thresholds ${CANDIDATE_CONFIDENCE_THRESHOLDS} \
    --step-estimator-refresh-unmasked-token-delta ${STEP_ESTIMATOR_REFRESH_UNMASKED_TOKEN_DELTA} \
    --step-estimate-update-interval ${STEP_ESTIMATE_UPDATE_INTERVAL} \
    --max-confidence-update-interval ${MAX_CONFIDENCE_UPDATE_INTERVAL} \
    --confidence-threshold-tput-demand-change-ratio ${CONFIDENCE_THRESHOLD_TPUT_DEMAND_CHANGE_RATIO} \
    --num-profile-runs ${NUM_PROFILE_RUNS} \
    --num-profile-warmup-runs ${NUM_PROFILE_WARMUP_RUNS} \
    --step-estimator-model-class ${STEP_ESTIMATOR_MODEL_CLASS} \
    --step-estimator-model-path ${STEP_ESTIMATOR_MODEL_PATH} \
    --step-estimator-features-path ${STEP_ESTIMATOR_FEATURES_PATH} \
    --step-data-dir ${STEP_DATA_DIR} \
    --reconfig-interval ${RECONFIG_INTERVAL} \
    --experiment-dir ${EXPERIMENT_DIR} \
    --total-num-requests ${total_num_requests} \
    --port ${VLLM_PORT} \
    --compilation-config '{\"full_cuda_graph\": true}'"

    if [ "$CACHE_PREFIX" = true ]; then
        vllm_cmd="$vllm_cmd --cache-prefix"
    fi
    if [ "$CACHE_SUFFIX" = true ]; then
        vllm_cmd="$vllm_cmd --cache-suffix"
    fi
    if [ "$SYNC_STEP_PREDICTION" = true ]; then
        vllm_cmd="$vllm_cmd --sync-step-prediction"
    fi
    if [ "$ENABLE_DROPPING" = true ]; then
        vllm_cmd="$vllm_cmd --enable-dropping"
    fi
    if [ -n "$MAX_NUM_UNFINISHED_REQUESTS" ]; then
        vllm_cmd="$vllm_cmd --max-num-unfinished-requests ${MAX_NUM_UNFINISHED_REQUESTS}"
    fi
    if [ "$DENOISE_BLOCK_SIZE" -gt -1 ]; then
        vllm_cmd="$vllm_cmd --denoise-block-size ${DENOISE_BLOCK_SIZE}"
    fi

    local eval_cmd
    eval_cmd="python eval/run_lmeval.py \
    --model $MODEL \
    --task $TASK \
    --output-length $OUTPUT_LENGTH \
    --output-path $output_path \
    --num-concurrent $NUM_CONCURRENT \
    --vllm-port $VLLM_PORT \
    --warmup"

    if [ -n "$ARRIVAL_PATTERN" ]; then
        eval_cmd="$eval_cmd --arrival-pattern \"$ARRIVAL_PATTERN\""
    fi
    if [ -n "$NUM_FEWSHOT" ]; then
        eval_cmd="$eval_cmd --num-fewshot $NUM_FEWSHOT"
    fi
    if [ "$WRITE_RESULTS" = true ]; then
        eval_cmd="$eval_cmd --write-results"
    fi

    echo "=== Running experiment: $experiment_name ==="
    echo "Running on host $(hostname) at path $(pwd)"
    echo "Config file: $CONFIG_FILE"
    echo "Batch dir: $SBATCH_DIR"
    echo "Experiment dir: $run_dir"

    echo "# Commands used for analysis" > "$run_dir/commands.txt"
    echo "$vllm_cmd" >> "$run_dir/commands.txt"
    echo "$eval_cmd" >> "$run_dir/commands.txt"

    export VLLM_LOGGING_LEVEL

    setsid bash --noprofile --norc -c "$vllm_cmd 2>&1 | tee \"$log_dir/vllm_serve.log\"" &
    local server_pid=$!
    CURRENT_SERVER_PID="$server_pid"

    wait_for_server "$VLLM_PORT" "$server_pid"

    run_logged "$eval_cmd" "$log_dir/lmeval_run.log"
    run_logged_allow_failure "python analysis/slo_attainment_and_good_accuracy/run.py --path \"$output_path\" --slo \"$SLO\"" "$log_dir/result_summary.log" "result summary"

    if [[ "$SCHEDULER_CLASS" == *TeDi* ]]; then
        run_logged_allow_failure "python analysis/confidence_over_time/plot.py --step-data \"$STEP_DATA_FILE\" --workload-history \"$WORKLOAD_FILE\" --output-dir \"$REQUEST_PLOTS_DIR\"" "$log_dir/confidence_over_time.log" "confidence-over-time plot"
    fi

    run_logged_allow_failure "python analysis/profiler_analysis/scheduler/run_schedule.py \"$run_dir/profiles/scheduler/schedule.jsonl\"" "$log_dir/schedule_analysis.log" "schedule analysis"
    run_logged_allow_failure "python analysis/calculate_num_recompute/run.py --path \"$run_dir/system_log.json\"" "$log_dir/num_recompute.log" "num recompute analysis"
    run_logged_allow_failure "python analysis/profiler_analysis/scheduler/run_update.py \"$run_dir/profiles/scheduler/update.jsonl\"" "$log_dir/update_analysis.log" "update analysis"
    run_logged_allow_failure "python analysis/profiler_analysis/model_runner/run.py \"$run_dir/profiles/model_runners/\"" "$log_dir/model_runner_analysis.log" "model runner analysis"
    run_logged_allow_failure "python analysis/profiler_analysis/step_estimator/run.py \"$run_dir/profiles/step_estimator/predict.jsonl\"" "$log_dir/step_estimator_analysis.log" "step estimator analysis"
    run_logged_allow_failure "python analysis/profiler_analysis/executor/run.py \"$run_dir/profiles/executors\"" "$log_dir/executor_analysis.log" "executor analysis"
    run_logged_allow_failure "python analysis/experiment_summary/run.py --result-path \"$output_path\" --slo \"$SLO\" --scheduler-summary \"$run_dir/profiles/scheduler/schedule_summary.txt\" --predict-summary \"$run_dir/profiles/step_estimator/predict_summary.txt\" --confidence-stats \"$REQUEST_PLOTS_DIR/avg_confidence_stats.json\"" "$log_dir/experiment_summary.log" "experiment summary"

    stop_server "$server_pid"
    CURRENT_SERVER_PID=""
    echo "=== Finished experiment: $experiment_name ==="
}

write_final_summary() {
    local summary_path="$SBATCH_DIR/final_summary.txt"

    python - "$summary_path" "${#EXPERIMENT_NAMES[@]}" "${EXPERIMENT_NAMES[@]}" "${EXPERIMENT_RUN_DIRS[@]}" <<'PY'
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
num_experiment_names = int(sys.argv[2])
args = sys.argv[3:]
experiment_names = args[:num_experiment_names]
experiment_run_dirs = [Path(path) for path in args[num_experiment_names:]]

metric_patterns = {
    "slo_attainment": r"SLO attainment \(≤.*?s\): ([0-9.]+)",
    "accuracy": r"Accuracy: ([0-9.]+)",
    "avg_response_time": r"Average response time: ([0-9.]+)s",
    "avg_scheduler_latency": r"Average scheduler overhead / latency: ([0-9.]+) ms",
    "document_number": r"Document number: ([0-9]+)",
    "predict_count": r"Predict count: ([0-9]+)",
    "predict_samples": r"Predict samples: ([0-9]+)",
    "avg_confidence": r"Average confidence: ([0-9.]+)",
}

rows = []
for name, run_dir in zip(experiment_names, experiment_run_dirs):
    log_path = run_dir / "logs" / "experiment_summary.log"
    if not log_path.is_file():
        rows.append((name, {"error": f"missing {log_path}"}))
        continue

    text = log_path.read_text()
    metrics = {}
    for key, pattern in metric_patterns.items():
        match = re.search(pattern, text)
        metrics[key] = match.group(1) if match else "N/A"
    rows.append((name, metrics))


def as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def as_int(value):
    try:
        return int(value)
    except Exception:
        return None


def fmt_float(value, width, precision):
    parsed = as_float(value)
    if parsed is None:
        return f"{'N/A':>{width}}"
    return f"{parsed:{width}.{precision}f}"


def fmt_int(value, width):
    parsed = as_int(value)
    if parsed is None:
        return f"{'N/A':>{width}}"
    return f"{parsed:{width}d}"

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w") as f:
    f.write("=== FINAL BATCH SUMMARY ===\n")
    f.write(f"experiments: {len(rows)}\n\n")
    f.write(f"{'experiment':30} | {'slo':>6} | {'acc':>6} | {'avg_rt(s)':>9} | {'sched(ms)':>10} | {'docs':>6} | {'pred_cnt':>8} | {'pred_samp':>9} | {'conf':>6}\n")
    f.write("-" * 120 + "\n")

    numeric_keys = ["slo_attainment", "accuracy", "avg_response_time", "avg_scheduler_latency", "document_number", "predict_count", "predict_samples", "avg_confidence"]
    aggregate = {key: [] for key in numeric_keys}

    for name, metrics in rows:
        if "error" in metrics:
            f.write(f"{name:30} | {'ERROR':>6} | {metrics['error']}\n")
            continue

        for key in numeric_keys:
            value = as_float(metrics.get(key, "N/A"))
            if value is not None:
                aggregate[key].append(value)

        f.write(
            f"{name:30} | {fmt_float(metrics.get('slo_attainment', 'N/A'), 6, 3)} | "
            f"{fmt_float(metrics.get('accuracy', 'N/A'), 6, 3)} | "
            f"{fmt_float(metrics.get('avg_response_time', 'N/A'), 9, 3)} | "
            f"{fmt_float(metrics.get('avg_scheduler_latency', 'N/A'), 10, 4)} | "
            f"{fmt_int(metrics.get('document_number', 'N/A'), 6)} | "
            f"{fmt_int(metrics.get('predict_count', 'N/A'), 8)} | "
            f"{fmt_int(metrics.get('predict_samples', 'N/A'), 9)} | "
            f"{fmt_float(metrics.get('avg_confidence', 'N/A'), 6, 3)}\n"
        )

    if rows:
        f.write("\n=== AVERAGES ACROSS EXPERIMENTS ===\n")
        for key in numeric_keys:
            values = aggregate[key]
            if not values:
                continue
            f.write(f"{key}: {sum(values) / len(values):.4f}\n")

print(summary_path)
PY
    echo "Final batch summary written to $summary_path"
}

echo "Config file: $CONFIG_FILE"
mapfile -t EXPERIMENT_NAMES < <(load_experiment_names "$CONFIG_FILE")

if [ ${#EXPERIMENT_NAMES[@]} -eq 0 ]; then
    echo "No experiments found in $CONFIG_FILE"
    exit 1
fi

start_ray_cluster

for index in "${!EXPERIMENT_NAMES[@]}"; do
    eval "$(load_config_exports "$CONFIG_FILE" "$index")"
    if [ -z "${EXPERIMENT_NAME:-}" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAMES[$index]}"
    fi

    run_one_experiment "$EXPERIMENT_NAME" "$index"
    write_final_summary
done

write_final_summary