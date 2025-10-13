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

from vllm.platforms import current_platform

from llm_proxy_server import launch_proxy
import logging

logging.basicConfig(
    level=logging.INFO,  # or INFO
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

SHOULD_APPLY_CHAT_TEMPLATE = {
    "GSAI-ML/LLaDA-8B-Instruct": True,
    "GSAI-ML/LLaDA-8B-Base": False,
}


def run_test(args):
    """Run the end to end accuracy test."""
    logger.info(f"Running test with args: {args}")
    real_base_url = "http://localhost:8000/v1"

    logger.info(f"launching proxy to {real_base_url}")
    launch_proxy(real_base_url, 
                 port=12345, 
                 apply_chat_template=SHOULD_APPLY_CHAT_TEMPLATE[args.model],
                 tokenizer_name=args.model,
                 avg_inter_arrival_time=args.avg_inter_arrival_time,
                 num_requests=args.limit,
                 output_path=args.output_path
                 )

    proxy_url = "http://localhost:12345/v1/completions"

    model_args = (
        f"model={args.model},"
        f"base_url={proxy_url},"
        f"num_concurrent={args.num_concurrent},"
        f"tokenized_requests=False,"
        f"timeout=10000")

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
    )

    if args.write_results:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)

    measured_value = results["results"][args.task]
    print(f"Measured value: {measured_value}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="Model name")
    parser.add_argument(
        "--task", type=str, help="Task name")
    parser.add_argument(
        "--limit", type=int, default=100, help="Limit the number of samples to eval"
    )
    parser.add_argument(
        "--output-length", type=int, help="Output length"
    )
    parser.add_argument(
        "--output-path", type=str, help="Output path"
    )
    parser.add_argument(
        "--num-concurrent", type=int, default=100, help="Number of concurrent connections"
    )
    parser.add_argument(
        "--avg-inter-arrival-time", type=float, default=0.0, help="Average inter-arrival time between requests in seconds"
    )
    parser.add_argument(
        "--write-results", action="store_true", help="Whether to write results to output path"
    )
    args = parser.parse_args()
    run_test(args)
    
if __name__ == "__main__":
    main()
