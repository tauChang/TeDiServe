import asyncio
import time
import json
import os
from typing import Any, Callable, Dict, Optional, Union

from vllm.logger import init_logger
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)

class FakeExecutor(Executor):
    """
    A fake executor that simulates model execution latency without running 
    actual computations. Used for measuring scheduler scaling and overhead.
    """

    uses_ray = False

    def _init_executor(self) -> None:
        self.latency_profiles: Dict[int, float] = {}
        self._load_latency_profile()
        logger.info("FakeExecutor initialized.")

    def collective_rpc(
        self,
        method: Union[str, Callable[..., Any]],
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        del timeout
        kwargs = kwargs or {}

        # FakeExecutor has no real workers. ExecutorBase.initialize_cache()
        # calls collective_rpc("initialize_cache"), so routing it back to self
        # causes infinite recursion.
        if method == "initialize_cache":
            return [None]

        if callable(method):
            result = method(self, *args, **kwargs)
        else:
            fn = getattr(self, method)
            result = fn(*args, **kwargs)

        return [result]


    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # Fake executor does not allocate real KV cache.
        logger.info(
            "FakeExecutor skipping KV cache allocation: # GPU blocks: %d, # CPU blocks: %d",
            num_gpu_blocks,
            num_cpu_blocks,
        )
        return None

    def check_health(self) -> None:
        # Nothing external to probe in fake mode.
        return None

    def _load_latency_profile(self):
        # dirname = self.vllm_config.profile_config.latency_profile_dir
        # model_name = self.vllm_config.model_config.model.replace("/", "_")
        # tp_degree = self.parallel_config.tensor_parallel_size
        
        # In a real setup, we'd detect this from ray/cluster config
        # accel = "nvidia_a100" 
        # profile_path = f"{dirname}/{model_name}/{accel}/TP{tp_degree}.json"
        profile_path = "/u/tchang85/dllm/latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP1.json"
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r") as f:
                    data = json.load(f)
                    self.latency_profiles = {int(k): float(v) for k, v in data.items()}
                logger.info(f"FakeExecutor loaded latency profile from {profile_path}")
            except Exception as e:
                logger.error(f"Failed to load latency profile: {e}")
        else:
            logger.warning(f"Latency profile not found at {profile_path}. Using 0ms latency.")

    def get_latency(self, num_tokens: int) -> float:
        if not self.latency_profiles:
            return 0.0
        sorted_keys = sorted(self.latency_profiles.keys())
        for k in sorted_keys:
            if k >= num_tokens:
                return self.latency_profiles[k]
        return self.latency_profiles[sorted_keys[-1]]

    def profile_latency(self) -> dict[int, float]:
        # ExecutorsManager persists this output to the latency profile path.
        return dict(self.latency_profiles)

    @staticmethod
    def _get_req_ids(scheduler_output) -> list[str]:
        # SchedulerOutput does not expose req_ids directly.
        return list(scheduler_output.num_scheduled_tokens.keys())

    @staticmethod
    def _build_confidence_stats(num_reqs: int) -> list[dict[str, float]]:
        # Keep parity with sampler output keys expected by Request.update.
        base_conf = {
            "output_avg": 1.0,
            "output_min": 1.0,
            "output_q25": 1.0,
            "output_median": 1.0,
            "output_q75": 1.0,
            # Optional aggregate fields consumed by some logging paths.
            "avg": 1.0,
            "min": 1.0,
            "q25": 1.0,
            "median": 1.0,
            "q75": 1.0,
        }
        return [dict(base_conf) for _ in range(num_reqs)]

    @staticmethod
    def _build_sampled_token_ids(scheduler_output, req_ids: list[str]) -> list[list[tuple[int, int]]]:
        sampled: list[list[tuple[int, int]]] = []
        logger.debug(f"scheduler_output: {scheduler_output}")
        logger.debug(f"req_ids: {req_ids}")
        for req_id in req_ids:
            logger.debug(f"Processing req_id {req_id}")
            # Use scheduler-provided exec start position to keep token position
            # compatible with request-side block range assertions.
            # assume one token unmasked per denoise step
            if req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                logger.debug(f"Req_id {req_id} found in scheduled_cached_reqs")
                # find the position of the request in scheduled_cached_reqs
                r_id = scheduler_output.scheduled_cached_reqs.req_ids.index(req_id)
                logger.debug(f"Req_id {req_id} has index {r_id} in scheduled_cached_reqs")
                pos = scheduler_output.scheduled_cached_reqs.cur_block_start[
                    r_id] + (scheduler_output.scheduled_cached_reqs.num_denoise_ran[r_id] % scheduler_output.scheduled_cached_reqs.denoise_block_size[r_id])
                logger.debug(f"Req_id {req_id} found in scheduled_cached_reqs with cur_block_start {scheduler_output.scheduled_cached_reqs.cur_block_start[r_id]} and num_denoise_ran {scheduler_output.scheduled_cached_reqs.num_denoise_ran[r_id]}")
            else:
                logger.debug(f"Req_id {req_id} not found in scheduled_cached_reqs, checking scheduled_new_reqs")
                pos = None
                # find the request in scheduled_new_reqs
                for rd in scheduler_output.scheduled_new_reqs:
                    if rd.req_id == req_id:
                        pos = rd.cur_block_start + (rd.num_denoise_ran % rd.denoise_block_size)
                        logger.debug(f"Req_id {req_id} found in scheduled_new_reqs with cur_block_start {rd.cur_block_start} and num_denoise_ran {rd.num_denoise_ran}")
                        break
                assert pos is not None, f"Req_id {req_id} not found in either scheduled_cached_reqs or scheduled_new_reqs"
            sampled.append([(pos, 42)])
            logger.debug(f"Built sampled token ids for req_id {req_id}: {sampled[-1]}")
        return sampled

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        latency = self.get_latency(num_tokens)
        logger.debug(f"Using latency {latency} for {num_tokens} tokens")
        if latency > 0:
            time.sleep(latency)

        req_ids = self._get_req_ids(scheduler_output)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=self._build_sampled_token_ids(scheduler_output, req_ids),
            confidence_stats=self._build_confidence_stats(len(req_ids)),
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            finished_sending=None,
            finished_recving=None,
        )

    async def execute_model_async(self, scheduler_output) -> ModelRunnerOutput:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        latency = self.get_latency(num_tokens)
        
        logger.debug(f"Using async latency {latency} for {num_tokens} tokens")
        if latency > 0:
            await asyncio.sleep(latency)

        req_ids = self._get_req_ids(scheduler_output)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=self._build_sampled_token_ids(scheduler_output, req_ids),
            confidence_stats=self._build_confidence_stats(len(req_ids)),
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            finished_sending=None,
            finished_recving=None,
        )

    def initialize_from_config(self, kv_cache_configs) -> None:
        pass

    def determine_available_memory(self) -> list[int]:
        return [85 * 1024 * 1024 * 1024]

    def get_kv_cache_specs(self) -> list[dict]:
        return [{}]

    def shutdown(self):
        pass
