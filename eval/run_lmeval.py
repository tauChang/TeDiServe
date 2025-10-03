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

from vllm.platforms import current_platform

# from ....utils import RemoteOpenAIServer
from llm_proxy_server import launch_proxy

# MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
# MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
NUM_CONCURRENT = 100
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
EXPECTED_VALUE = 0.54
# DEFAULT_ARGS = ["--max-model-len", "4096", "--disable-log-requests"]
DEFAULT_ARGS = ["--trust-remote-code",
                "--distributed-executor-backend", "ray", 
                "--num-gpus-per-model-executor", "1", 
                "--scheduler_cls", "vllm.v1.core.sched.cluster_scheduler.ClusterScheduler", 
                "--denoise-block-size", "32", 
                "--cache-prefix", 
                "--cache-suffix"
]
MORE_ARGS_LIST = [
    # [],  # Default
    # ["--enable-chunked-prefill"],  # Chunked
    # ["--num-scheduler-steps", "8"],  # MS
    # ["--num-scheduler-steps", "8", "--multi-step-stream-outputs"]  # MS+Stream
    # ["--max-tokens", "128"]
    []
]
SHOULD_APPLY_CHAT_TEMPLATE = {
    "GSAI-ML/LLaDA-8B-Instruct": True,
    "GSAI-ML/LLaDA-8B-Base": False,
}


def run_test(more_args):
    """Run the end to end accuracy test."""
    prefix = False
    suffix = False
    block_size = 32
    confidence = 0.9
    output_length = 256

    # args = list(DEFAULT_ARGS)
    # args.extend(more_args)
    # print(f"Running with: {args}")

    # Launch proxy to sit in front of the actual server
    real_base_url = "http://localhost:8000/v1"
    launch_proxy(real_base_url, port=12345, 
                 apply_chat_template=SHOULD_APPLY_CHAT_TEMPLATE[MODEL_NAME],
                 tokenizer_name=MODEL_NAME
                 )

    proxy_url = "http://localhost:12345/v1/completions"

    model_args = (
        f"model={MODEL_NAME},"
        f"base_url={proxy_url},"
        f"num_concurrent={NUM_CONCURRENT},"
        f"tokenized_requests=False,"
        f"timeout=10000")

    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=TASK,
        write_out=True,
        log_samples=True,
        verbosity="INFO",
        gen_kwargs={
            "max_tokens": output_length,
        },
        limit=100
    )

    

    # RESULTS IS A DICT. SAVE AS JSON
    import json
    with open(f"results{'_prefix' if prefix else ''}{'_suffix' if suffix else ''}_block{block_size}_conf{confidence}_out{output_length}_instruct.json", "w") as f:
        json.dump(results, f, indent=2)

    measured_value = results["results"][TASK][FILTER]
    print(f"Measured value: {measured_value}")
    # assert (measured_value - RTOL < EXPECTED_VALUE
    #         and measured_value + RTOL > EXPECTED_VALUE
    #         ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"


@pytest.mark.skipif(not current_platform.is_cuda()
                    and not current_platform.is_tpu(),
                    reason="V1 currently only supported on CUDA and TPU")
def test_lm_eval_accuracy_v1_engine(monkeypatch: pytest.MonkeyPatch):
    """Run with the V1 Engine."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        more_args = []

        # Limit compilation time for V1
        if current_platform.is_tpu():
            # more_args = ["--max-num-seqs", "64"]
            more_args = ["--max-num-seqs", "10"]

        run_test(more_args)


# @pytest.mark.parametrize("more_args", MORE_ARGS_LIST)
# def test_lm_eval_accuracy_v0_engine(monkeypatch: pytest.MonkeyPatch,
#                                     more_args):
#     """Run with the V0 Engine."""

#     with monkeypatch.context() as m:
#         m.setenv("VLLM_USE_V1", "0")
#         run_test(more_args)
