import json

# ---- common config ----
common = {
    "VLLM_LOGGING_LEVEL": "INFO",
    "MODEL": "GSAI-ML/LLaDA-8B-Instruct",
    "NUM_GPUS_PER_MODEL_EXECUTOR": 1,
    "SCHEDULER_CLASS": "vllm.v1.core.sched.tedi_new_correct_tput_async_update_scheduler.TeDiLightScheduler",
    "DEFAULT_CONFIDENCE_THRESHOLD": 0.9,
    "CANDIDATE_CONFIDENCE_THRESHOLDS": ["0.9", "0.8", "0.7", "0.6", "0.5"],
    "CONFIDENCE_THRESHOLD_TPUT_DEMAND_CHANGE_RATIO": 100000000,
    "NUM_PROFILE_RUNS": 8,
    "NUM_PROFILE_WARMUP_RUNS": 3,
    "TASK": "gsm8k",
    "OUTPUT_LENGTH": 256,
    "WRITE_RESULTS": True,
    "VLLM_PORT": 8083,
    "DENOISE_BLOCK_SIZE": 32,
    "CACHE_PREFIX": True,
    "CACHE_SUFFIX": True,
    "STEP_ESTIMATOR_REFRESH_UNMASKED_TOKEN_DELTA": 1000,
    "SYNC_STEP_PREDICTION": False,
    "STEP_ESTIMATOR_FEATURES_PATH": "./analysis/denoise_step_prediction/models/lgb/sharegpt_0415_all_features/features.txt",
    "NUM_CONCURENT": 200
}

# ---- model configs ----
model_configs = {
    "oracle": {
        "STEP_ESTIMATOR_MODEL_CLASS": "vllm.v1.core.sched.step_estimator.models.oracle.Oracle",
        "STEP_ESTIMATOR_MODEL_PATH": "./analysis/gsm8k_oracle/req_denoise_counts_by_conf.json",
    },
    "lightgbm": {
        "STEP_ESTIMATOR_MODEL_CLASS": "vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine.LightGradientBoostMachine",
        "STEP_ESTIMATOR_MODEL_PATH": "./analysis/denoise_step_prediction/models/lgb/sharegpt_0415_all_features/model.bin",
    },
}

# ---- parameters ----
rps_values = [10, 12, 14, 16]
slo_values = [10, 8, 6]

# ---- generate experiments ----
experiments = []
for model_name, model_cfg in model_configs.items():
    for slo in slo_values:
        for rps in rps_values:
            experiments.append({
                "name": f"{model_name}_slo_{slo}_rps_{rps}",
                "SLO": slo,
                "ARRIVAL_PATTERN": f"1319:{rps}",  # adjust if needed
                **model_cfg,
            })

# ---- final config ----
config = {
    "common": common,
    "experiments": experiments,
}

# ---- write to file ----
output_file = "oracle_vs_one_shot.json"
with open(output_file, "w") as f:
    json.dump(config, f, indent=2)

print(f"Wrote {len(experiments)} experiments to {output_file}")