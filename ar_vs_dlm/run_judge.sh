#!/bin/bash
set -e

BASE="/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251130"

for TS in 152122 203923 231147; do
    INPUT_FILE="${BASE}/${TS}/logs/benchmark_latency_info.json"
    OUTPUT_FILE="${BASE}/${TS}/logs/judge_results.json"

    echo "Running judge on ${TS}..."
    python run_judge.py --input "${INPUT_FILE}" --output "${OUTPUT_FILE}" 2>&1 | tee "${BASE}/${TS}/logs/judge_run.log"
done
