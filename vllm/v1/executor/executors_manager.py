# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from typing import Optional, Union

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.executor.ray_distributed_executor import (  # noqa
    RayDistributedExecutor as RayDistributedExecutorV0)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config,
                                         unify_kv_cache_configs)
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput

import json
import os
import time
import threading
from pathlib import Path

from typing import Any, Callable, Optional, TypeVar, Union
_R = TypeVar("_R")

logger = init_logger(__name__)


def get_latency_profile_path(vllm_config: VllmConfig,
                             bundle_ids: list[int]) -> Optional[str]:
    dirname = vllm_config.profile_config.latency_profile_dir
    model_name = vllm_config.model_config.model.replace("/", "_")
    denoise_block_size = vllm_config.model_config.denoise_block_size

    accelerator_type = None
    print(f"bundle_specs: {vllm_config.cluster_config.placement_group.bundle_specs}")
    for bundle in vllm_config.cluster_config.placement_group.bundle_specs:
        for key in bundle:
            print(f"key: {key}")
            if key.startswith("accelerator_type:"):
                accelerator_type = key.split(":")[1]
                break
        if accelerator_type is not None:
            break

    assert accelerator_type is not None
    
    tp_degree = len(bundle_ids)

    return f"{dirname}/{model_name}_block{denoise_block_size}/{accelerator_type}/TP{tp_degree}.json"


class ExecutorsManager:
    def __init__(self, 
                 executor_class: type[Executor], 
                 vllm_config: VllmConfig,
                 executor_fail_callback: Optional[callable] = None
                 ):
        self.executor_class = executor_class
        self.vllm_config = vllm_config
        self.executor_fail_callback = executor_fail_callback

        self.executors: dict[int, Executor] = {}
        self.cond: dict[int, asyncio.Condition] = {}

        self.used_executor_ids = set()

        self.lock = threading.Lock()
        # a lock for updating these
        self.scheduler_kv_cache_config = {}
        self.num_cpu_blocks = {}
        self.num_gpu_blocks = {}
    
    def __repr__(self) -> str:
        return (
            f"ExecutorsManager("
            f"executors={self.executors}, "
            f"used_executor_ids={self.used_executor_ids})"
        )
    
    async def launch_executor(self, 
                              executor_id: int,
                              bundle_ids: list[int]
                              ) -> None:
        assert executor_id not in self.used_executor_ids

        logger.debug(f"Launching executor {executor_id} with bundles {bundle_ids}")
        executor = await asyncio.to_thread(
            self.executor_class, self.vllm_config, executor_id, bundle_ids)
        logger.debug(f"Executor {executor_id} created.")

        num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config = \
            await asyncio.to_thread(self.initialize_kv_caches, executor)
        
        latency_profile_path = get_latency_profile_path(
            self.vllm_config, bundle_ids)

        if not os.path.exists(latency_profile_path):
            logger.info(f"Profiling latency for executor {executor_id}")
            # touch it so other executors know it's being profiled
            os.makedirs(os.path.dirname(latency_profile_path), exist_ok=True)
            Path(latency_profile_path).touch()
            try:
                profile = executor.collective_rpc("profile_latency", args=())
                profile = profile[0]
                with open(latency_profile_path, "w") as f:
                    json.dump(profile, f, indent=4)

                logger.info(f"Latency profile saved to {latency_profile_path}")
            except Exception as e:
                os.remove(latency_profile_path)
                raise e

        if self.executor_fail_callback is not None:
            executor.register_failure_callback(self.executor_fail_callback)
        
        self.executors[executor_id] = executor
        self.cond[executor_id] = asyncio.Condition()
        
        self.used_executor_ids.add(executor_id)
        self.num_gpu_blocks[executor_id] = num_gpu_blocks
        self.num_cpu_blocks[executor_id] = num_cpu_blocks
        self.scheduler_kv_cache_config[executor_id] = \
            scheduler_kv_cache_config

        logger.debug(f"Executor {executor_id} launched.")
    
    async def kill_executor(self, executor_id) -> None:
        # assume exectutors_manager lock is held
        logger.debug(f"Killing executor {executor_id}")

        assert executor_id in self.used_executor_ids
        executor = self.executors[executor_id]
        
        async with self.cond[executor_id]:
            logger.debug(f"waiting for executor {executor_id} to be idle")
            await self.cond[executor_id].wait_for(
                lambda: executor.is_idle())
            logger.debug(f"start shutting down executor {executor_id}. Status transition: IDLE -> KILLING")
            self.executors[executor_id].set_killing()
            await asyncio.to_thread(self.executors[executor_id].shutdown)

        del self.executors[executor_id]
        del self.cond[executor_id]
            
        logger.debug(f"Executor {executor_id} killed.")
    
    def collective_rpc(self, 
                       executor_id: int, 
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        
        return self.executors[executor_id].collective_rpc(method, timeout, 
                                                                args, kwargs)
    
    def initialize_kv_caches(self, executor):
        start = time.time()
        logger.debug(f"Initializing kv caches for executor {executor.id}")
        
        # Get all kv cache needed by the model
        kv_cache_specs = executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                # self.available_gpu_memory_for_kv_cache = \
                #     ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                available_gpu_memory = [
                    # self.available_gpu_memory_for_kv_cache
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                ] * len(kv_cache_specs)
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = (
                    executor.determine_available_memory())
                # self.available_gpu_memory_for_kv_cache = \
                #     available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)
        # Get the kv cache tensor size
        kv_cache_configs = [
            get_kv_cache_config(self.vllm_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip(kv_cache_specs, available_gpu_memory)
        ]

        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        unify_kv_cache_configs(kv_cache_configs)

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        assert all([
            cfg.num_blocks == kv_cache_configs[0].num_blocks
            for cfg in kv_cache_configs
        ])
        num_gpu_blocks = kv_cache_configs[0].num_blocks
        num_cpu_blocks = 0
        scheduler_kv_cache_config = kv_cache_configs[0]

        # Initialize kv cache and warmup the execution
        executor.initialize_from_config(kv_cache_configs)

        # with self.lock:
        #     self.num_gpu_blocks[executor_id] = num_gpu_blocks
        #     self.num_cpu_blocks[executor_id] = num_cpu_blocks
        #     self.scheduler_kv_cache_config[executor_id] = \
        #         scheduler_kv_cache_config
        executor.collective_rpc("initialize_cache", 
                                args = (num_gpu_blocks, num_cpu_blocks))

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return num_cpu_blocks, num_gpu_blocks, scheduler_kv_cache_config
    
    def get_scheduler_kv_cache_config(self, executor_id):
        assert executor_id in self.scheduler_kv_cache_config
        return self.scheduler_kv_cache_config[executor_id]
        
                                                            
    def shutdown(self):
        for executor in self.executors.values():
            executor.shutdown()
    
    def profile(self, is_start: bool):
        for executor in self.executors.values():
            if executor.profile:
                executor.profile(is_start)
        
    def sleep(self, level: int = 1):
        for executor in self.executors.values():
            executor.sleep(level)
    
    def wake_up(self, tags: Optional[list[str]] = None):
        for executor in self.executors.values():
            executor.wake_up(tags)
    
    def is_sleeping(self) -> bool:
        for executor in self.executors.values():
            if not executor.is_sleeping():
                return False
        return True