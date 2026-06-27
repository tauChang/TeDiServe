# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file test accuracy of the vLLM server via LMEval.
It uses local-completions, which interacts with vLLM
through the OAI API with N concurrent connections.
This simulates real work usage of the API and makes
sure that the zmq frontend mp RPC message passing and
AsyncLLMEngine are working correctly.
"""

import lm_eval
import pytest
import argparse
import json
import os
import psutil
import numpy as np


from vllm.platforms import current_platform

from llm_proxy_server import launch_proxy
# from llm_proxy_server_mbpp import launch_proxy
import logging
import time

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

logging.basicConfig(
    level=logging.INFO,  # or INFO
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_NUM_FEWSHOT_BY_TASK = {
    "gsm8k": 5,
    "mmlu_pro": 0,
    "mbpp": 3,
    "mbpp_instruct": 3,
}

SHOULD_APPLY_CHAT_TEMPLATE = {
    "GSAI-ML/LLaDA-8B-Instruct": True,
    "GSAI-ML/LLaDA-8B-Base": False,
    "Dream-org/Dream-v0-Instruct-7B": True,
    "Dream-org/Dream-v0-Base-7B": False,
}

def get_slurm_assigned_cpus():
    affinity_override = os.environ.get("CPU_AFFINITY")
    if affinity_override:
        cpu_list = []
        for part in affinity_override.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = map(int, part.split("-"))
                cpu_list.extend(range(start, end + 1))
            else:
                cpu_list.append(int(part))
        return cpu_list

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        try:
            return list(psutil.Process().cpu_affinity())
        except AttributeError:
            return list(range(psutil.cpu_count(logical=True) or 1))

    uid = os.getuid()
    path = f"/sys/fs/cgroup/cpuset/slurm/uid_{uid}/job_{job_id}/cpuset.cpus"

    try:
        with open(path, "r") as f:
            cpus = f.read().strip()
    except FileNotFoundError:
        cps = os.cpu_count()
        cpus = f"0-{cps-1}"
        # raise RuntimeError(f"SLURM cpuset file not found: {path}")

    # expand ranges like 0-15,32-47 → [0,1,2,...15,32,...47]
    cpu_list = []
    for part in cpus.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            cpu_list.extend(range(start, end + 1))
        else:
            cpu_list.append(int(part))

    return cpu_list

# def parse_arrival_pattern(pattern_str: str):
#     """
#     Parse a string like '50:1.0,50:3.0' into a list of (num_requests, inter_arrival_time) tuples.
#     """
#     phases = []
#     for phase in pattern_str.split(","):
#         if not phase.strip():
#             continue
#         num_str, interval_str = phase.split(":")
#         phases.append((int(num_str.strip()), float(interval_str.strip())))
#     return phases
def parse_arrival_pattern(pattern_str: str, task: str, default_cv: float = 1.0):
    """
    Parse a string like:
        '50:10, 50:5'
        '50:10:2.0, 100:5'

    where each segment is:
        num_requests:rps[:cv]

    Returns a list of:
        (num_requests, mean_interarrival, cv)

    mean_interarrival is computed as 1 / rps.
    If cv is omitted, default_cv is used.
    """
    phases = []
    for phase in pattern_str.split(","):
        phase = phase.strip()
        if not phase:
            continue

        parts = phase.split(":")
        if len(parts) == 2:
            # Format: num:rps
            num, rps = parts
            cv = default_cv
        elif len(parts) == 3:
            # Format: num:rps:cv
            num, rps, cv = parts
        else:
            raise ValueError(
                f"Invalid arrival pattern segment '{phase}'. "
                f"Expected num:rps or num:rps:cv"
            )

        rps = float(rps)
        if rps <= 0:
            raise ValueError(
                f"Invalid RPS '{rps}' in segment '{phase}'. RPS must be > 0."
            )

        mean_interarrival = 1.0 / rps

        if task == "mmlu_pro":
            num = str(int(num) * 14)  # scale up for mmlu_pro

        phases.append((int(num), mean_interarrival, float(cv)))

    return phases

def remove_fewshot_samples(results):
    # print(f"before removal: {results}")
    configs = results.get("configs", {})
    for task, cfg in configs.items():
        # print(f"looking at task: {task}, cfg: {cfg}")
        few = cfg.get("fewshot_config")
        if isinstance(few, dict) and "samples" in few:
            del few["samples"]
    # print(f"after removal: {results}")

import functools
import types

# def clean_configs_for_json(results):
#     """
#     Remove or sanitize LM Eval config fields that contain non-serializable
#     objects such as functions, functools.partial, or callables.
#     This preserves all metrics and task results.
#     """
#     cfgs = results.get("configs", {})
    
#     for task, cfg in cfgs.items():
#         keys_to_delete = []

#         for k, v in cfg.items():
#             # Remove functions or functools.partial
#             if isinstance(v, (types.FunctionType, functools.partial)):
#                 keys_to_delete.append(k)

#             # Nested fewshot_config
#             if k == "fewshot_config" and isinstance(v, dict):
#                 nested_delete = []
#                 for fk, fv in v.items():
#                     if isinstance(fv, (types.FunctionType, functools.partial)):
#                         nested_delete.append(fk)
#                 for fk in nested_delete:
#                     del v[fk]

#             # generation_kwargs and filter_list are usually OK
            
#         # delete top-level fields
#         for k in keys_to_delete:
#             del cfg[k]

def clean_configs_for_json(results):
    """
    Remove or sanitize LM Eval config fields that contain non-serializable
    objects such as functions, functools.partial, or callables.
    This preserves all metrics and task results.
    """
    cfgs = results.get("configs", {})
    
    for task, cfg in cfgs.items():
        keys_to_delete = []

        for k, v in cfg.items():
            # Remove functions or functools.partial at top level
            if isinstance(v, (types.FunctionType, functools.partial)):
                keys_to_delete.append(k)

            # Nested fewshot_config
            if k == "fewshot_config" and isinstance(v, dict):
                nested_delete = []
                for fk, fv in v.items():
                    if isinstance(fv, (types.FunctionType, functools.partial)):
                        nested_delete.append(fk)
                for fk in nested_delete:
                    del v[fk]

            # Clean filter_list: remove filter_fn
            if k == "filter_list" and isinstance(v, list):
                for flt in v:
                    if "filter" in flt and isinstance(flt["filter"], list):
                        for item in flt["filter"]:
                            if isinstance(item, dict) and "filter_fn" in item:
                                del item["filter_fn"]

        # Delete top-level fields
        for k in keys_to_delete:
            del cfg[k]


def get_num_fewshot(args):
    if args.num_fewshot is not None:
        return args.num_fewshot
    return DEFAULT_NUM_FEWSHOT_BY_TASK.get(args.task)


def run_test(args):
    """Run the end to end accuracy test."""
    logger.info(f"Running test with args: {args}")
    real_base_url = f"http://localhost:{args.vllm_port}/v1"

    logger.info(f"launching proxy to {real_base_url}")

    if args.arrival_pattern:
        arrival_pattern = parse_arrival_pattern(args.arrival_pattern, args.task)
        # args.limit = sum(num for num, _ in arrival_pattern)
        args.limit = sum(num for num, _, _ in arrival_pattern)
        if args.task == "mmlu_pro":
            args.limit = args.limit // 14
        args.limit = min(args.limit, 2000) # avoid overload the proxy server
    else:
        arrival_pattern = [(args.limit, args.avg_inter_arrival_time)]
    logger.info(f"Using arrival pattern: {arrival_pattern}")
    
    launch_proxy(real_base_url, 
                 port=12345, 
                 apply_chat_template=SHOULD_APPLY_CHAT_TEMPLATE[args.model],
                 tokenizer_name=args.model,
                 arrival_pattern=arrival_pattern,
                 output_path=args.output_path,
                 warmup=args.warmup
                 )

    proxy_url = "http://localhost:12345/v1/completions"

    model_args = (
        f"model={args.model},"
        f"trust_remote_code=True,"
        f"base_url={proxy_url},"
        f"num_concurrent={args.num_concurrent},"
        f"tokenized_requests=False,"
        f"timeout=10000")

    num_fewshot = get_num_fewshot(args)

    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=args.task,
        write_out=True,
        log_samples=True,
        verbosity="INFO",
        gen_kwargs={
            "max_tokens": args.output_length,
        },
        limit=args.limit,
        random_seed=0,
        confirm_run_unsafe_code=True,
        num_fewshot=num_fewshot,
    )

    # print(results)

    if args.write_results:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        if args.task == "gsm8k":
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)
        elif args.task in ["mbpp", "mbpp_instruct"]:
            # remove_fewshot_samples(results)
            clean_configs_for_json(results)
            # print(f"Results after removing fewshot samples: {results}")
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)
        elif args.task == "mmlu_pro":
            clean_configs_for_json(results)
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)

    measured_value = results["results"][args.task]
    print(f"Measured value: {measured_value}")
    time.sleep(5)  # wait for all profiler data to be flushed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="Model name")
    parser.add_argument(
        "--task", type=str, help="Task name")
    parser.add_argument(
        "--limit", type=int, help="Limit the number of samples to eval"
    )
    parser.add_argument(
        "--num-concurrent", type=int, help="Number of concurrent connections"
    )
    parser.add_argument(
        "--avg-inter-arrival-time", type=float, help="Average inter-arrival time between requests in seconds"
    )
    parser.add_argument(
        "--arrival-pattern",
        type=str,
        help="Arrival pattern in 'num:rps,num:rps;...' format (optional cv: 'num:rps:cv'). "
            "Example: '50:10,50:5' means first 50 req at 10 RPS, next 50 at 5 RPS."
    )
    parser.add_argument(
        "--output-length", type=int, help="Output length"
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Override the number of few-shot examples. If omitted, task-specific defaults are used.",
    )
    parser.add_argument(
        "--output-path", type=str, help="Output path"
    )
    parser.add_argument(
        "--vllm-port", type=int, default=8000, help="Port where the vLLM server is listening"
    )
    parser.add_argument(
        "--write-results", action="store_true", help="Whether to write results to output path"
    )
    parser.add_argument(
        "--warmup", action="store_true", help="Whether to perform warmup requests before starting evaluation"
    )
    args = parser.parse_args()

    # --- Argument validation ---
    if args.arrival_pattern is not None:
        assert args.limit is None and args.avg_inter_arrival_time is None, \
            "--arrival-pattern cannot be used with --limit or --avg-inter-arrival-time"
    else:
        assert args.limit is not None and args.avg_inter_arrival_time is not None, \
            "Both --limit and --avg-inter-arrival-time must be specified when --arrival-pattern is not provided."

    
    run_test(args)
    
if __name__ == "__main__":
    p = psutil.Process()
    all_cpus = get_slurm_assigned_cpus()
    p.cpu_affinity(all_cpus)
    logger.info("Using CPU affinity: %s", all_cpus)
    main()
