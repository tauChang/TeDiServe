#!/bin/bash
#SBATCH --job-name=dllm-judge
#SBATCH --output=sbatch_experiment_dir/slurm-%x-%j.out
#SBATCH --error=sbatch_experiment_dir/slurm-%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=ghx4-interactive
#SBATCH --account=bftv-dtai-gh
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=72
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tchang85@wisc.edu

set -euo pipefail

STORAGE_DIR=${STORAGE_DIR:-/work/hdd/bftv/tchang85}
SCRIPT_DIR=${SCRIPT_DIR:-/u/tchang85/dllm/llm_judge}
PYTHON_BIN=${PYTHON_BIN:-/u/tchang85/miniconda3/envs/vllm/bin/python}
VLLM_BIN=${VLLM_BIN:-/u/tchang85/miniconda3/envs/vllm/bin/vllm}

MODEL=${MODEL:-meta-llama/Llama-3.3-70B-Instruct}
PORT=${PORT:-8007}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
DTYPE=${DTYPE:-bfloat16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
VLLM_LOGGING_LEVEL=DEBUG

DEFAULT_EXPERIMENT_DIRS=(
    "/u/tchang85/dllm/sbatch_experiment_dir/20260503/163209_sharegpt_20_llumnix_only/0_llumnix_20qps"
    "/u/tchang85/dllm/sbatch_experiment_dir/20260503/193754_sharegpt_20_infaas_only/0_infaas_20qps"
)

if [ -n "${EXPERIMENT_DIRS:-}" ]; then
    read -r -a EXPERIMENT_DIR_ARRAY <<< "$EXPERIMENT_DIRS"
else
    EXPERIMENT_DIR_ARRAY=("${DEFAULT_EXPERIMENT_DIRS[@]}")
fi

cd "$SCRIPT_DIR"

DATE_DIR=$(date +"%Y%m%d")
TIME_DIR=$(date +"%H%M%S")
SBATCH_DIR="$STORAGE_DIR/sbatch_experiment_dir/$DATE_DIR/${TIME_DIR}_judge"
mkdir -p "$SBATCH_DIR"

exec > >(tee -a "$SBATCH_DIR/stdout.log") 2> >(tee -a "$SBATCH_DIR/stderr.log" >&2)

ln -sfn "$SBATCH_DIR" /u/tchang85/dllm/current_experiment
cp "$SCRIPT_DIR/run_judge_sbatch.sh" "$SBATCH_DIR/"

SERVER_LOG="$SBATCH_DIR/vllm_server.log"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -- -"$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_server() {
    local timeout_seconds=${1:-600}
    local start_time
    start_time=$(date +%s)

    echo "Waiting for judge server on port ${PORT}..."
    while true; do
        if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
            return 0
        fi

        if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Judge server exited before becoming ready" >&2
            tail -n 200 "$SERVER_LOG" >&2 || true
            return 1
        fi

        if (( $(date +%s) - start_time >= timeout_seconds )); then
            echo "Timed out waiting for judge server" >&2
            tail -n 200 "$SERVER_LOG" >&2 || true
            return 1
        fi

        sleep 2
    done
}

echo "Starting judge server"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"

export VLLM_LOGGING_LEVEL

setsid "$VLLM_BIN" serve "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    > >(tee -a "$SERVER_LOG") 2> >(tee -a "$SERVER_LOG" >&2) &
SERVER_PID=$!

wait_for_server 600

for EXPERIMENT_DIR in "${EXPERIMENT_DIR_ARRAY[@]}"; do
    LOG_DIR="${EXPERIMENT_DIR}/logs"
    INPUT_FILE="${LOG_DIR}/benchmark_latency_info.json"
    OUTPUT_FILE="${LOG_DIR}/judge_results.json"
    RUN_LOG="${LOG_DIR}/judge_run.log"

    mkdir -p "$LOG_DIR"
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Skipping ${EXPERIMENT_DIR}: missing input ${INPUT_FILE}" >&2
        continue
    fi

    echo "Running judge on ${EXPERIMENT_DIR}..."
    "$PYTHON_BIN" run_judge.py --input "$INPUT_FILE" --output "$OUTPUT_FILE" 2>&1 | tee "$RUN_LOG"
done

echo "Judge batch completed"