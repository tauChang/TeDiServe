#!/usr/bin/env python3
"""Run the TeDiServe Artifact Evaluation demonstration.

``smoke`` mode runs the real TeDiServe scheduler with a simulated executor.
It is self-contained: it does not load model weights, require a GPU, contact
Hugging Face, or use Gurobi. ``gpu`` mode launches the regular
TeDiServe server with LLaDA and submits a small concurrent workload.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifact"
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_SCHEDULER = (
    "vllm.v1.core.sched.tediserve_scheduler.TeDiServeScheduler"
)
DEFAULT_ESTIMATOR = (
    "vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine."
    "LightGradientBoostMachine"
)

# Short GSM8K-style questions make the optional GPU run inspectable.  They are
# intentionally simple and bundled here, so the evaluator does not need to
# download a benchmark dataset.  The expected answers are diagnostics rather
# than a performance or accuracy claim.
DEMO_QUESTIONS = (
    ("Mia has 7 apples and buys 5 more. How many apples does she have?", "12"),
    ("A box holds 6 pencils. How many pencils are in 4 boxes?", "24"),
    ("Noah read 9 pages on Monday and 8 pages on Tuesday. How many pages total?", "17"),
    ("A train travels 3 miles each hour for 5 hours. How many miles does it travel?", "15"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("auto", "smoke", "gpu"), default="auto",
        help="Execution resource to use (default: auto-detect GPU availability).")
    parser.add_argument(
        "--instances", type=int, default=None,
        help="Number of model instances. Default: 2 for smoke mode; available "
        "GPU count capped at 2 for gpu mode.")
    parser.add_argument(
        "--requests", type=int, default=8,
        help="Number of short requests to submit (default: 8).")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for logs and demo_result.json.")
    parser.add_argument(
        "--hf-home", type=Path, default=None,
        help="Hugging Face cache directory (GPU mode; default: output-dir/hf-cache).")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Public Hugging Face model for gpu mode.")
    parser.add_argument(
        "--port", type=int, default=None,
        help="Local port for gpu mode. Default: choose a free port.")
    parser.add_argument(
        "--startup-timeout", type=int, default=1200,
        help="Maximum seconds to wait for the GPU server (default: 1200).")
    return parser.parse_args()


def default_output_dir(mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "artifact-results" / f"{stamp}-{mode}"


def configure_process_environment(output_dir: Path, *, smoke: bool,
                                  hf_home: Path | None = None) -> None:
    # Do not inherit a site-wide cache blindly: it may be inaccessible on a
    # compute node or contain credentials for another environment. Evaluators
    # can opt into a shared cache explicitly with --hf-home.
    os.environ["HF_HOME"] = str(hf_home or (output_dir / "hf-cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib"))
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    if smoke:
        # This opt-in is handled only by fake-backend branches in vLLM.
        os.environ["VLLM_FAKE_EXECUTOR_CPU"] = "1"
        os.environ.setdefault("VLLM_FAKE_ACCELERATOR", "GH200")
        os.environ["VLLM_FAKE_PROFILE_MODEL"] = \
            "GSAI-ML_LLaDA-8B-Instruct"


def empty_kv_cache_config() -> Any:
    from vllm.v1.kv_cache_interface import KVCacheConfig

    return KVCacheConfig(num_blocks=4096, kv_cache_tensors=[],
                         kv_cache_groups=[])


async def run_smoke_demo(instances: int, requests: int,
                         output_dir: Path) -> dict[str, Any]:
    """Run the TeDiServe scheduler with simulated model execution."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.tediserve_scheduler import \
        TeDiServeScheduler
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.executor.executors_manager import ExecutorsManager
    from vllm.v1.executor.fake_executor import FakeExecutor
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

    # The trailing comma keeps this an explicit list of instances; a bare
    # "1" would mean one-GPU instances on every visible GPU.
    instance_spec = ",".join("1" for _ in range(instances)) + ","
    estimator_dir = REPO_ROOT / "step_estimator_models" / "lgb_gsm8k"
    engine_args = AsyncEngineArgs(
        model=str(ARTIFACT_DIR / "smoke_model"),
        skip_tokenizer_init=True,
        distributed_executor_backend="fake",
        num_gpus_per_model_executor=instance_spec,
        load_format="dummy",
        disable_async_output_proc=True,
        scheduler_cls=DEFAULT_SCHEDULER,
        request_latency_slo=10,
        default_confidence_threshold=0.9,
        candidate_confidence_thresholds=[0.9, 0.8, 0.7],
        step_estimator_refresh_unmasked_token_delta=8,
        confidence_threshold_tput_demand_change_ratio=0.1,
        step_estimator_model_class=DEFAULT_ESTIMATOR,
        step_estimator_model_path=str(estimator_dir / "model.bin"),
        step_estimator_features_path=str(estimator_dir / "features.txt"),
        step_data_dir=str(output_dir),
        experiment_dir=str(output_dir),
        total_num_requests=requests,
        reconfig_interval=-1,
        cache_prefix=False,
        cache_suffix=False,
        latency_profile_dir=str(REPO_ROOT / "latency_profiles"),
    )
    vllm_config = engine_args.create_engine_config()
    manager = ExecutorsManager(FakeExecutor, vllm_config)
    kv_cache_config = empty_kv_cache_config()

    for executor_id in range(instances):
        manager.executors[executor_id] = FakeExecutor(
            vllm_config, executor_id, [executor_id])
        manager.scheduler_kv_cache_config[executor_id] = kv_cache_config
        manager.cond[executor_id] = asyncio.Condition()
        manager.waiting_to_be_killed[executor_id] = False
        manager.used_executor_ids.add(executor_id)

    scheduler = TeDiServeScheduler(
        vllm_config, manager, StructuredOutputManager(vllm_config),
        log_stats=False)
    assignments = {str(executor_id): 0 for executor_id in range(instances)}
    completed: list[str] = []

    try:
        for request_id in range(requests):
            request = EngineCoreRequest(
                request_id=f"demo-{request_id}",
                prompt_token_ids=[1, 2, 3, 4],
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=SamplingParams(temperature=0.0,
                                               max_tokens=6),
                pooling_params=None,
                eos_token_id=2,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            )
            scheduler.add_request(Request.from_engine_core_request(
                request,
                mask_token_id=vllm_config.model_config.mask_token_id,
                latency_slo=vllm_config.model_config.request_latency_slo,
                denoise_block_size=vllm_config.model_config.denoise_block_size,
            ))

        # Six simulated denoising steps are sufficient for each request. A generous
        # bound keeps failures explicit if a future scheduler change regresses
        # the demonstration.
        for _ in range(requests * 12):
            outputs = await scheduler.schedule()
            for executor_id, output in outputs.items():
                if not output.total_num_scheduled_tokens:
                    continue
                assignments[str(executor_id)] += len(
                    output.num_scheduled_tokens)
                model_output = await manager.executors[executor_id] \
                    .execute_model_async(output)
                engine_outputs = await scheduler.update_from_output(
                    executor_id, output, model_output)
                for output_group in engine_outputs.values():
                    completed.extend(
                        item.request_id for item in output_group.outputs
                        if item.finished)
            if not scheduler.has_requests():
                break
    finally:
        scheduler.shutdown()

    completed = sorted(set(completed))
    used_instances = sum(value > 0 for value in assignments.values())
    passed = len(completed) == requests and used_instances == instances
    return {
        "mode": "smoke",
        "backend": "TeDiServe simulated executor",
        "instances": instances,
        "requests_submitted": requests,
        "requests_completed": len(completed),
        "completed_request_ids": completed,
        "per_instance_assignments": assignments,
        "used_instances": used_instances,
        "passed": passed,
        "note": (
            "No model weights or GPU were used. The real TeDiServe scheduler "
            "latency profile. This validates scheduling/control plumbing, "
            "not model quality or meaningful confidence-threshold adaptation."),
    }


def gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def choose_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(port: int, process: subprocess.Popen[str],
                    timeout_seconds: int) -> None:
    endpoint = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The GPU server exited before becoming ready.")
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for TeDiServe on port {port}.")


def submit_gpu_request(port: int, model: str, request_id: int) -> dict[str, Any]:
    question, expected_answer = DEMO_QUESTIONS[request_id % len(DEMO_QUESTIONS)]
    body = json.dumps({
        "model": model,
        "prompt": (
            "Solve this elementary arithmetic question. Reply with only the "
            "integer answer.\n\nQuestion: " + question + "\nAnswer:"),
        "max_tokens": 64,
        "temperature": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
        payload["_artifact_http_status"] = response.status
        payload["_artifact_request_index"] = request_id
        payload["_artifact_question"] = question
        payload["_artifact_expected_answer"] = expected_answer
        return payload


def summarize_gpu_response(response: dict[str, Any]) -> dict[str, Any]:
    """Retain small, evaluator-readable evidence without copying full output."""
    choices = response.get("choices") or []
    first_choice = choices[0] if choices else {}
    text = first_choice.get("text", "")
    if not isinstance(text, str):
        text = ""
    return {
        "request_index": response.get("_artifact_request_index"),
        "http_status": response.get("_artifact_http_status"),
        "response_id": response.get("id"),
        "choices": len(choices),
        "finish_reason": first_choice.get("finish_reason"),
        "text_characters": len(text),
        "text_preview": text.replace("\n", "\\n")[:160],
        "question": response.get("_artifact_question"),
        "expected_answer": response.get("_artifact_expected_answer"),
        "contains_expected_answer": str(
            response.get("_artifact_expected_answer", "")) in text,
    }


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_gpu_demo(instances: int, requests: int, output_dir: Path,
                 model: str, port: int, startup_timeout: int) -> dict[str, Any]:
    available_gpus = gpu_count()
    if available_gpus < instances:
        raise RuntimeError(
            f"GPU mode requested {instances} instance(s), but only "
            f"{available_gpus} GPU(s) are visible. Use --mode smoke or lower "
            "--instances.")

    estimator_dir = REPO_ROOT / "step_estimator_models" / "lgb_gsm8k"
    # The trailing comma keeps this an explicit list of instances; a bare
    # "1" would mean one-GPU instances on every visible GPU.
    instance_spec = ",".join("1" for _ in range(instances)) + ","
    # Ray workers in separate TeDiServe executors need distinct rendezvous
    # ports. Supplying a checked-free base port makes concurrent executor
    # startup deterministic; vLLM reserves subsequent ports under its lock.
    rendezvous_port = choose_free_port()
    server_log = output_dir / "server.log"
    command = [
        "vllm", "serve", "--trust-remote-code", model,
        "--distributed-executor-backend", "ray",
        "--num-gpus-per-model-executor", instance_spec,
        "--scheduler-cls", DEFAULT_SCHEDULER,
        "--request-latency-slo", "30",
        "--default-confidence-threshold", "0.9",
        "--candidate-confidence-thresholds", "0.9", "0.8", "0.7",
        "--step-estimator-refresh-unmasked-token-delta", "8",
        "--confidence-threshold-tput-demand-change-ratio", "0.1",
        "--step-estimator-model-class", DEFAULT_ESTIMATOR,
        "--step-estimator-model-path", str(estimator_dir / "model.bin"),
        "--step-estimator-features-path", str(estimator_dir / "features.txt"),
        "--latency-profile-dir", str(REPO_ROOT / "latency_profiles"),
        "--step-data-dir", str(output_dir),
        "--experiment-dir", str(output_dir),
        "--total-num-requests", str(requests),
        "--reconfig-interval", "-1",
        # The small artifact run does not need CUDA-graph capture.  Eager
        # execution also avoids requiring a separately generated capture-size
        # list on a fresh evaluator checkout.
        "--enforce-eager",
        "--port", str(port),
    ]
    (output_dir / "command.txt").write_text(" ".join(command) + "\n")

    with server_log.open("w") as log_file:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, text=True, stdout=log_file,
            stderr=subprocess.STDOUT, start_new_session=True,
            # Existing scheduler debug records identify each executor that
            # accepts a request. Keep them for the small artifact run so the
            # JSON result can verify cross-instance scheduling.
            env={**os.environ,
                 "VLLM_LOGGING_LEVEL": "DEBUG",
                 "VLLM_PORT": str(rendezvous_port)})
        try:
            wait_for_server(port, process, startup_timeout)
            responses: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=requests) as pool:
                futures = [pool.submit(submit_gpu_request, port, model, i)
                           for i in range(requests)]
                for future in as_completed(futures):
                    responses.append(future.result())
        finally:
            terminate_process_group(process)

    response_summaries = sorted(
        (summarize_gpu_response(response) for response in responses),
        key=lambda item: item["request_index"],
    )
    completed = sum(bool(response.get("choices")) for response in responses)
    nonempty_completions = sum(
        item["text_characters"] > 0 for item in response_summaries)
    expected_answer_matches = sum(
        item.get("contains_expected_answer", False)
        for item in response_summaries)
    log_text = server_log.read_text(errors="replace")
    assignments = {
        str(executor_id): len(re.findall(
            rf"Executor {executor_id} adding request", log_text))
        for executor_id in range(instances)
    }
    used_instances = sum(value > 0 for value in assignments.values())
    return {
        "mode": "gpu",
        "backend": "TeDiServe GPU executors",
        "model": model,
        "instances": instances,
        "visible_gpus": available_gpus,
        "rendezvous_port": rendezvous_port,
        "requests_submitted": requests,
        "requests_completed": completed,
        "nonempty_completions": nonempty_completions,
        "expected_answer_matches": expected_answer_matches,
        "response_summaries": response_summaries,
        "per_instance_assignments": assignments,
        "used_instances": used_instances,
        "passed": (
            completed == requests
            and used_instances == instances
            and expected_answer_matches == requests
        ),
        "note": (
            "This mode loads the public LLaDA model and performs real GPU "
            "execution. Dynamic reconfiguration is disabled, so Gurobi is "
            "not required."),
    }


def main() -> int:
    args = parse_args()
    if args.requests < 2:
        raise SystemExit("--requests must be at least 2 to demonstrate scheduling.")
    if args.instances is not None and args.instances < 1:
        raise SystemExit("--instances must be positive.")

    mode = args.mode
    available_gpus = gpu_count()
    if mode == "auto":
        mode = "gpu" if available_gpus else "smoke"
    instances = args.instances or (min(available_gpus, 2) if mode == "gpu" else 2)
    output_dir = (args.output_dir or default_output_dir(mode)).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    configure_process_environment(output_dir, smoke=mode == "smoke",
                                  hf_home=args.hf_home)

    started = time.monotonic()
    if mode == "smoke":
        result = asyncio.run(run_smoke_demo(instances, args.requests, output_dir))
    else:
        result = run_gpu_demo(
            instances, args.requests, output_dir, args.model,
            args.port or choose_free_port(), args.startup_timeout)
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (output_dir / "demo_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    if result["passed"]:
        print(f"TeDiServe artifact demo PASS ({mode} mode).")
        return 0
    print(f"TeDiServe artifact demo FAIL ({mode} mode).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
