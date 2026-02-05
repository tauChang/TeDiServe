#!/bin/bash
TP=4,
CACHE=True

BASE_DIR=./tp_experiment_dir
DATE_DIR=$(date +"%Y%m%d")
TIME_DIR=$(date +"%H%M%S")
EXPERIMENT_DIR="${BASE_DIR}/${DATE_DIR}/${TIME_DIR}"
export VLLM_TORCH_PROFILER_DIR=${EXPERIMENT_DIR}/torch_profiler

# mkdir
mkdir -p ${EXPERIMENT_DIR}
mkdir -p ${VLLM_TORCH_PROFILER_DIR}


VLLM_CMD="vllm serve \
  --trust-remote-code \
  GSAI-ML/LLaDA-8B-Instruct \
  --distributed-executor-backend ray \
  --num-gpus-per-model-executor $TP \
  --scheduler_cls vllm.v1.core.sched.infaas_aligned_scheduler.InFaaSAlignedScheduler \
  --default-confidence-threshold 1.1 \
  --num-profile-runs 8 \
  --num-profile-warmup-runs 3 \
  --step-estimator-model-class vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine.LightGradientBoostMachine \
  --step-estimator-model-path ./analysis/denoise_step_prediction/models/lgb/no_cache_256_32/model.bin \
  --step-estimator-features-path ./analysis/denoise_step_prediction/models/lgb/no_cache_256_32/features.txt \
  --step-data-dir ${EXPERIMENT_DIR} \
  --experiment-dir ${EXPERIMENT_DIR} \
  --total-num-requests 100 \
  --denoise-block-size 32 \
  --compilation-config {\"full_cuda_graph\":true} \
  "

if [ "$CACHE" = "True" ]; then
  VLLM_CMD="${VLLM_CMD} --cache-prefix --cache-suffix"
fi

# run command
echo $VLLM_CMD
$VLLM_CMD 2>&1 | tee ${EXPERIMENT_DIR}/serve_log.txt