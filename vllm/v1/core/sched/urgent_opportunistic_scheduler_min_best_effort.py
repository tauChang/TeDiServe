# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import bisect
from collections import defaultdict, OrderedDict
from collections.abc import Iterable
import heapq
import itertools
import json
import os
import time
import copy
from typing import Any, Optional, Union
from enum import Enum, IntEnum

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.executor.executor_base import ExecutorStatus
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.step_estimator import StepEstimator, StepStats
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop, LatencyProfile, SystemLogger
from vllm.v1.utils import TimeProfiler
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.executors_manager import ExecutorsManager, get_latency_profile_path
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)

class OrderedSet:
    def __init__(self, iterable: Optional[Iterable[Any]] = None) -> None:
        self._dict = OrderedDict()
        if iterable is not None:
            for item in iterable:
                self._dict[item] = None

    def add(self, item: Any) -> None:
        self._dict[item] = None

    def discard(self, item: Any) -> None:
        self._dict.pop(item, None)

    def __contains__(self, item: Any) -> bool:
        return item in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        items = ", ".join(repr(item) for item in self._dict)
        return f"OrderedSet([{items}])"

class ExecutorState:
    def __init__(self, executor: Executor, token_budget: int) -> None:
        self.executor = executor
        self.executor_id =  executor.id
        self.tp_degree = len(executor.bundle_ids)
        self.req_ids: OrderedSet[str] = OrderedSet()
        self.pending_request_additions: OrderedSet[str] = OrderedSet()
        self.pending_request_removals: OrderedSet[str] = OrderedSet()
        self.req_to_tokens_needed: dict[str, int] = {}
        self._token_budget = token_budget
        # TODO
        # self.is_urgent = False
    
    def __repr__(self) -> str:
        return (f"ExecutorState(tp_degree={self.tp_degree}, "
                f"req_ids={self.req_ids}, "
                f"pending_request_additions={self.pending_request_additions}, "
                f"pending_request_removals={self.pending_request_removals}, "
                f"req_to_tokens_needed={self.req_to_tokens_needed}, "
                f"token_budget={self._token_budget})")
    
    def add_request(self, request: Request) -> None:
        req_id = request.request_id
        logger.debug(f"Executor {self.executor_id} adding request {req_id}.")
        self.req_ids.add(req_id)
        self.req_to_tokens_needed[req_id] = request.num_tokens

    def remove_request(self, req_id: str) -> None:
        logger.debug(f"Executor {self.executor_id} removing request {req_id}.")
        self.req_ids.discard(req_id)
        self.req_to_tokens_needed.pop(req_id)
        if req_id in self.pending_request_removals:
            self.unmark_pending_request_removal(req_id)
    
    def mark_pending_request_addition(self, request: Request) -> None:
        req_id = request.request_id
        logger.debug(f"Executor {self.executor_id} marking pending addition of request {req_id}.")
        self.pending_request_additions.add(req_id)
        self.req_to_tokens_needed[req_id] = request.num_tokens
    
    def mark_pending_request_removal(self, req_id: str) -> None:
        logger.debug(f"Executor {self.executor_id} marking pending removal of request {req_id}.")
        self.pending_request_removals.add(req_id)
    
    def unmark_pending_request_addition(self, req_id: str) -> None:
        logger.debug(f"Executor {self.executor_id} unmarking pending addition of request {req_id}.")
        assert req_id in self.pending_request_additions
        self.pending_request_additions.discard(req_id)
        self.req_to_tokens_needed.pop(req_id, None)
    
    def unmark_pending_request_removal(self, req_id: str) -> None:
        logger.debug(f"Executor {self.executor_id} unmarking pending removal of request {req_id}.")
        assert req_id in self.pending_request_removals
        self.pending_request_removals.discard(req_id)
    
    def commit_pending_request_addition(self, req_id: str) -> None:
        logger.debug(f"Executor {self.executor_id} committing pending addition of request {req_id}.")
        assert req_id in self.pending_request_additions
        assert req_id not in self.req_ids
        self.pending_request_additions.discard(req_id)
        self.req_ids.add(req_id)
    
    def commit_pending_request_removal(self, req_id: str) -> None:
        logger.debug(f"Executor {self.executor_id} committing pending removal of request {req_id}.")
        assert req_id in self.pending_request_removals
        assert req_id in self.req_ids
        self.pending_request_removals.discard(req_id)
        self.req_ids.discard(req_id)
        self.req_to_tokens_needed.pop(req_id)

    def get_projected_req_ids(self) -> list[str]:
        req_ids = [r_id for r_id in self.req_ids if r_id not in self.pending_request_removals]
        req_ids.extend(self.pending_request_additions)
        return req_ids
                
    def get_batch_size(self) -> int:
        return sum(
            self.req_to_tokens_needed[req_id] for req_id in self.req_ids)
    
    def get_projected_batch_size(self) -> int:
        return sum(
            self.req_to_tokens_needed[req_id] for req_id in self.get_projected_req_ids()
        )

    def get_remaining_budget(self) -> int:
        return self._token_budget - self.get_batch_size()
    
    def get_projected_remaining_budget(self) -> int:
        return self._token_budget - self.get_projected_batch_size()
    
    def has_no_pending_changes(self) -> bool:
        return len(self.pending_request_additions) == 0 and \
                len(self.pending_request_removals) == 0
    
    def sorted_req_ids(self, request_states: dict[str, RequestState]) -> list[str]:
        # sort by priority
        # normal > opportunistic > best_effort
        # tie breaker: num_tokens (decreasing)
        def req_sort_key(req_id: str) -> tuple[int, int]:
            # req_state.priority must not be None since they're on this executor
            # return (priority, num_tokens)
            return (request_states[req_id].priority.value, -self.req_to_tokens_needed[req_id])
        return sorted(self.req_ids, key=req_sort_key)
    
    def get_conditional_req_ids(self, request_states: dict[str, RequestState], conditions: set[RequestStatePriority]) -> list[str]:
        return [r_id for r_id in self.req_ids if request_states[r_id].priority in conditions]
    
    def get_normal_req_ids(self, request_states: dict[str, RequestState]) -> list[str]:
        return [r_id for r_id in self.req_ids if request_states[r_id].is_normal]
    
    def get_opportunistic_req_ids(self, request_states: dict[str, RequestState]) -> list[str]:
        return [r_id for r_id in self.req_ids if request_states[r_id].is_opportunistic]
    
    def get_best_effort_req_ids(self, request_states: dict[str, RequestState]) -> list[str]:
        return [r_id for r_id in self.req_ids if request_states[r_id].is_best_effort]
    
    def is_drained(self) -> bool:
        return len(self.req_ids) == 0 and \
                len(self.pending_request_additions) == 0 and \
                len(self.pending_request_removals) == 0
    
    # def is_urgent(self, request_states: dict[str, RequestState]) -> bool:
    #     for req_id in self.req_ids:
    #         if request_states[req_id].is_urgent:
    #             return True
    #     return False
    
    # def is_projected_urgent(self, request_states: dict[str, RequestState]) -> bool:
    #     for req_id in self.get_projected_req_ids():
    #         if request_states[req_id].pending_is_urgent:
    #             return True
    #     return False
                
# enum RequestStateStatus: UNSCHEDULED, SCHEDULED, RUNNING
class RequestStateStatus(Enum):
    UNSCHEDULED = 0
    SCHEDULED = 1
    RUNNING = 2

class RequestStatePriority(IntEnum):
    NORMAL = 0
    OPPORTUNISTIC = 1
    BEST_EFFORT = 2

class RequestState:
    def __init__(self, request: Request) -> None:
        self.request_id = request.request_id if request is not None else None
        self.executor_id = None
        self.executors_to_free: OrderedSet[int] = OrderedSet()
        self.pending_executor_id = None # executor to be set in the next scheduling step
        self.pred_num_steps_left = {} # confidence_threshold -> predicted steps left
        self.last_stats: StepStats = StepStats(
            id = self.request_id,
            timestamp="",
            num_denoise_ran=0,
            num_unmasked_tokens=0,
            num_cur_unmasked_tokens=0,
            output_length=request.output_length,
            block=0,
            block_num_denoise_ran=0,
            block_num_unmasked_tokens=0,
            block_size=request.denoise_block_size,
            confidence_threshold=0.0,
            max_confidence_threshold=0.0,
            last_recompute_avg_output_confidence=0.0,
            cur_avg_output_confidence=0.0,
        ) if request is not None else None # for copy
        # self.is_urgent = None
        self.priority = None
        self.confidence_threshold = None
        # self.pending_is_urgent = None
        self.pending_priority = None
        self.pending_confidence_threshold = None
        self.status: RequestStateStatus = RequestStateStatus.UNSCHEDULED
        self.max_confidence_threshold_idx = 0 
    
    def __deepcopy__(self, memo) -> RequestState:
        new_request_state = RequestState(request=None)
        new_request_state.request_id = self.request_id
        new_request_state.executor_id = self.executor_id
        new_request_state.executors_to_free = None #copy.deepcopy(self.executors_to_free, memo)
        new_request_state.pending_executor_id = self.pending_executor_id
        new_request_state.pred_num_steps_left = None #copy.deepcopy(self.pred_num_steps_left, memo)
        new_request_state.last_stats = None #copy.deepcopy(self.last_stats, memo)
        # new_request_state.is_urgent = self.is_urgent
        new_request_state.priority = self.priority
        new_request_state.confidence_threshold = self.confidence_threshold
        # new_request_state.pending_is_urgent = self.pending_is_urgent
        new_request_state.pending_priority = self.pending_priority
        new_request_state.pending_confidence_threshold = self.pending_confidence_threshold
        new_request_state.status = self.status
        return new_request_state
    
    def __str__(self) -> str:
        return (f"RequestState(request_id={self.request.request_id}, "
                f"executor_id={self.executor_id}, "
                f"executors_to_free={self.executors_to_free}, "
                f"pending_executor_id={self.pending_executor_id}, "
                f"last_stats={self.last_stats})")
    
    def set_executor(self, executor_id: int, priroity: RequestStatePriority, confidence_threshold: float) -> None:
        # logger.debug(f"Request {self.request.request_id} setting executor to {executor_id}")
        self.executor_id = executor_id
        if executor_id is not None:
            self.executors_to_free.add(executor_id)
        self.priority = priroity
        self.confidence_threshold = confidence_threshold

        self.refresh_status()
        # logger.debug(f"after setting, request state: {self}")
    
    def remove_executor(self) -> None:
        # logger.debug(f"Request {self.request.request_id} clearing executor {self.executor_id}")
        self.executors_to_free.discard(self.executor_id)
        self.executor_id = None
        self.priority = None
        self.confidence_threshold = None

        self.refresh_status()
        # logger.debug(f"after clearing, request state: {self}")
    
    def set_pending_executor(self, executor_id: int, priority: RequestStatePriority, confidence_threshold: float) -> None:
        assert executor_id is not None
        self.pending_executor_id = executor_id
        # self.pending_is_urgent = is_urgent
        self.pending_priority = priority
        self.pending_confidence_threshold = confidence_threshold

        self.refresh_status()
        # logger.debug(f"after setting pending, request state: {self}")
    
    def remove_pending_executor(self) -> None:
        self.pending_executor_id = None
        # self.pending_is_urgent = None
        self.pending_priority = None
        self.pending_confidence_threshold = None

        self.refresh_status()
    
    def commit_pending_executor(self) -> None:
        self.set_executor(self.pending_executor_id, self.pending_priority, self.pending_confidence_threshold)
        # pending_executor_id, pending_is_urgent, pending_confidence_threshold remain the same

        self.refresh_status()

    def remove_executor_to_free(self, executor_id: int) -> None:
        # logger.debug(f"Request {self.request.request_id} removing executor to free {executor_id}")
        self.executors_to_free.discard(executor_id)
        # logger.debug(f"after removing, request state: {self}")
    
    @property
    def is_unscheduled(self) -> bool:
        return self.status == RequestStateStatus.UNSCHEDULED
    
    @property
    def is_scheduled(self) -> bool:
        return self.status == RequestStateStatus.SCHEDULED
    
    @property
    def is_running(self) -> bool:
        return self.status == RequestStateStatus.RUNNING
    
    @property
    def is_normal(self) -> bool:
        return self.priority == RequestStatePriority.NORMAL
    
    @property
    def is_opportunistic(self) -> bool:
        return self.priority == RequestStatePriority.OPPORTUNISTIC
    
    @property
    def is_best_effort(self) -> bool:
        return self.priority == RequestStatePriority.BEST_EFFORT
    
    def refresh_status(self) -> None:
        if self.executor_id is not None:
            self.status = RequestStateStatus.RUNNING
        elif self.pending_executor_id is not None:
            self.status = RequestStateStatus.SCHEDULED
        else:
            self.status = RequestStateStatus.UNSCHEDULED


class UrgentOpportunisticScheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        executors_manager: ExecutorsManager,
        # kv_cache_configs: dict[int, KVCacheConfig],
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        # self.kv_cache_configs = kv_cache_configs
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        # self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
        #     defaultdict(set) if include_finished_set else None)
        self.finished_req_ids_dict: Optional[
            dict[int, dict[int, set[str]]]
            ] = defaultdict(lambda: defaultdict(set)) \
                if include_finished_set else None
        
        self.executors_manager = executors_manager
        self.executor_states: dict[int, ExecutorState] = {}
        self.max_tp_degree = 0

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs # assume this is per executor
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector_v1(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        # num_gpu_blocks = self.cache_config.num_gpu_blocks
        # for executor_id, num_blocks in num_gpu_blocks.items():
        #     assert num_blocks is not None and num_blocks > 0
        # assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        # # req_id -> Request
        # self.requests: dict[str, Request] = {}
        # # self.request_to_executor: dict[str, int] = {}

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # help keep track of additional scheduler states for requests
        self.request_states: dict[str, RequestState] = {}
        # self.request_to_executor: dict[str, int] = {}


        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        # self.unscheduled: list[str] = []
        # self.scheduled: list[str] = []
        # self.running: list[str] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        # self.finished_req_ids: set[str] = set()
        self.finished_req_ids: dict[int, set[str]] = defaultdict(set)
        self.free_req_ids: dict[int, set[str]] = defaultdict(set)

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config

        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        # [tau_chang] one kv_cache_manager per executor for now
        self.kv_cache_managers: dict[int, KVCacheManager] = {}
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

        self.default_confidence_threshold = \
            self.scheduler_config.default_confidence_threshold
        self.candidate_confidence_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
        self.best_effort_confidence_threshold = 0.5
        
        self.cache_prefix = vllm_config.model_config.cache_prefix
        self.cache_suffix = vllm_config.model_config.cache_suffix

        self.step_estimator = StepEstimator(vllm_config)
        self.latency_profiles: dict[int, LatencyProfile] = {}
        # read in latency profiles
        for tp_degree in [1, 2, 4]:
            path = get_latency_profile_path(vllm_config, tp_degree)
            self.latency_profiles[tp_degree] = LatencyProfile(path)
        self.throughput_supply: float = 0
        self.max_throughput_demand_per_req: float = 0
        
        self.system_logger = SystemLogger("system_logs", self)
        self.system_logger.log()

        profiler_path = os.path.join(
            self.vllm_config.experiment_config.experiment_dir,
            "schedule_profiler.jsonl")
        self.schedule_profiler = TimeProfiler("Schedule", profiler_path)

    def update_step_estimates(self, req_id: str) -> None:
        req = self.request_states[req_id]
        stats_with_confidence = []
        for conf in self.candidate_confidence_thresholds:
            stats = StepStats(
                id = req_id,
                timestamp=req.last_stats.timestamp,
                num_denoise_ran=req.last_stats.num_denoise_ran,
                num_unmasked_tokens=req.last_stats.num_unmasked_tokens,
                num_cur_unmasked_tokens=req.last_stats.num_cur_unmasked_tokens,
                output_length=req.last_stats.output_length,
                block=req.last_stats.block,
                block_num_denoise_ran=req.last_stats.block_num_denoise_ran,
                block_num_unmasked_tokens=req.last_stats.block_num_unmasked_tokens,
                block_size=req.last_stats.block_size,
                confidence_threshold=conf,
                max_confidence_threshold=req.last_stats.max_confidence_threshold,
                last_recompute_avg_output_confidence=req.last_stats.last_recompute_avg_output_confidence,
                cur_avg_output_confidence=req.last_stats.cur_avg_output_confidence,
            )
            stats_with_confidence.append(stats)
        
        req.pred_num_steps_left = {
            conf: step for conf, step in zip(
                self.candidate_confidence_thresholds, 
                self.step_estimator.predict(stats_with_confidence))
        }
            
        logger.debug(f"Updated step estimates for request {req_id}: {req.pred_num_steps_left} steps left")
    
    def get_profile_latency(self, tp_degree: int, batch_size: int) -> float:
        ceiling_batch_size, per_step_latency = self.latency_profiles[tp_degree].lookup(batch_size)
        # logger.debug(f"Profile latency for tp_degree {tp_degree} and batch_size {batch_size}: "
        #              f"ceiling_batch_size={ceiling_batch_size}, per_step_latency={per_step_latency}")
        return ceiling_batch_size, per_step_latency
    
    def get_min_time_left(self, req_id: str):
        # single step min time left
        batch_size = self.requests[req_id].num_tokens
        # logger.debug(f"max_tp_degree: {self.max_tp_degree}, batch_size: {batch_size}")
        _, min_time_left = self.get_profile_latency(self.max_tp_degree, batch_size)
        return min_time_left
    
    def is_possible_to_meet_slo(self, req_id: str):
        pred_remaining_time = self.get_min_time_left(req_id) * self.request_states[req_id].pred_num_steps_left[self.candidate_confidence_thresholds[-1]]
        remaining_time_lower_bound = pred_remaining_time * 0.7 # assume 30% error
        logger.debug(f"Request {req_id} possible to meet SLO check: "
                     f"slo_time_remaining={self.requests[req_id].slo_time_remaining}, "
                     f"pred_remaining_time={pred_remaining_time}, "
                     f"remaining_time_lower_bound={remaining_time_lower_bound}")
        return self.requests[req_id].slo_time_remaining > remaining_time_lower_bound
        # return self.requests[req_id].slo_time_remaining > self.get_min_time_left(req_id) * self.request_states[req_id].pred_num_steps_left[
    
    # def get_estimated_time_left(self, req_id: str, tp_degree: int, batch_size: int, confidence: float) -> float:
    #     pred_num_steps_left = self.request_states[req_id].pred_num_steps_left[confidence]
    #     ceiling_batch_size, per_step_latency = self.get_profile_latency(tp_degree, batch_size)
    #     # logger.debug(f"Estimated time left for request {req_id} with tp_degree {tp_degree} and batch_size {batch_size}: "
    #     #              f"confidence={confidence}, "
    #     #              f"pred_num_steps_left={pred_num_steps_left}, "
    #     #              f"per_step_latency={per_step_latency}, "
    #     #              f"ceiling_batch_size={ceiling_batch_size}")
    #     result = pred_num_steps_left * per_step_latency
    #     logger.debug(f"Estimated time left for request {req_id} with tp_degree {tp_degree} and batch_size {batch_size}: {result} seconds")
    #     return result

    def get_num_steps_left_by_type(self, req_id: str, confidence: float) -> int:
        pred_num_steps_left = self.request_states[req_id].pred_num_steps_left[confidence]
        if not self.cache_prefix and not self.cache_suffix:
            num_recompute_steps_left = pred_num_steps_left
        elif self.cache_prefix and not self.cache_suffix:
            num_recompute_steps_left = self.requests[req_id].num_recompute_steps_left
        elif self.cache_prefix and self.cache_suffix:
            num_recompute_steps_left = self.requests[req_id].num_recompute_steps_left
        num_cache_steps_left = max(0, pred_num_steps_left - num_recompute_steps_left)
        return num_recompute_steps_left, num_cache_steps_left

    def get_estimated_time_left(self, 
                                req_id: str, 
                                tp_degree: int, 
                                all_req_ids: Iterable[str],
                                confidence: float
                                ) -> float:
        """
        Assumptions: 
        When this request recomputes, all other requests also recompute.
        This is a pessimistic estimate since other requests can leave earlier.
        """
        recompute_batch_size = sum(self.requests[r_id].num_tokens for r_id in all_req_ids)
        _, recompute_step_latency = self.get_profile_latency(tp_degree, recompute_batch_size)
        num_recompute_steps_left, num_cache_steps_left = self.get_num_steps_left_by_type(req_id, confidence)

        if not self.cache_prefix and not self.cache_suffix:
            cache_batch_size = 0
            cache_step_latency = 0
        elif self.cache_prefix and not self.cache_suffix:
            cache_batch_size = sum(self.requests[r_id].output_length for r_id in all_req_ids)
            _, cache_step_latency = self.get_profile_latency(tp_degree, cache_batch_size)
        elif self.cache_prefix and self.cache_suffix:
            cache_batch_size = self.requests[req_id].denoise_block_size * len(all_req_ids)
            _, cache_step_latency = self.get_profile_latency(tp_degree, cache_batch_size)
        
        result = (num_recompute_steps_left * recompute_step_latency) + \
                    (num_cache_steps_left * cache_step_latency)
        logger.debug(f"Estimated time left for request {req_id} with tp_degree {tp_degree} and all_req_ids {all_req_ids}: ")
        logger.debug(f"recompute_batch_size={recompute_batch_size}, recompute_step_latency={recompute_step_latency}, num_recompute_steps_left={num_recompute_steps_left}")
        logger.debug(f"cache_batch_size={cache_batch_size}, cache_step_latency={cache_step_latency}, num_cache_steps_left={num_cache_steps_left}")
        
        return result
    
    def update_request_max_confidence_thresholds(self) -> None:
        return
        def get_tput_demand(r_id: str, confidence: float) -> float:
            num_recompute_steps_left, num_cache_steps_left = self.get_num_steps_left_by_type(r_id, confidence)
            tput_demand = (num_recompute_steps_left * self.requests[r_id].num_tokens + \
                num_cache_steps_left * self.requests[r_id].denoise_block_size) / self.requests[r_id].slo_time_remaining
            logger.debug(f"unbounded tput demand {tput_demand}")
            logger.debug(f"slo_time_remaining {self.requests[r_id].slo_time_remaining}")
            logger.debug(f"max tput_demand_per_req {self.max_throughput_demand_per_req}")
            tput_demand = min(tput_demand, self.max_throughput_demand_per_req)
            return tput_demand

        # go in decreasing progress order
        reqs_by_progress = sorted(
            self.request_states.keys(),
            key=lambda r_id: self.requests[r_id].unmask_progress,
            reverse=True,
        )
        n = len(reqs_by_progress)
        
        # compute cum_min_tput_suffix (from i+1 to end)
        # cum_min_tput_suffix[i] = sum of min tput demand of reqs from i+1 to end
        # cum_min_tput_suffix[n-1] = 0, cause no reqs after n-1
        cum_min_tput_suffix = [0.0] * n
        for i in range(n - 2, -1, -1):
            req_id = reqs_by_progress[i+1]
            req_state = self.request_states[req_id]
            if req_state.is_best_effort or not self.is_possible_to_meet_slo(req_id):
                min_tput_demand = 0.0
            else:
                min_tput_demand = get_tput_demand(req_id, self.candidate_confidence_thresholds[-1])
            cum_min_tput_suffix[i] = cum_min_tput_suffix[i + 1] + min_tput_demand
        
        logger.debug(f"cum_min_tput_suffix: {cum_min_tput_suffix}")
        
        cum_tput_demand = 0.0
        for i, req_id in enumerate(reqs_by_progress):
            logger.debug(f"Looking at the {i}-th request {req_id}")
            req_state = self.request_states[req_id]
            if req_state.is_best_effort or not self.is_possible_to_meet_slo(req_id):
                req_state.max_confidence_threshold_idx = 0 # highest confidence
                # req_state.max_confidence_threshold_idx = len(self.candidate_confidence_thresholds) - 1 # lowest confidence
                logger.debug(f"  best effort or cannot meet SLO, set max confidence to {self.candidate_confidence_thresholds[0]}")
                continue
            
            for idx, conf in enumerate(self.candidate_confidence_thresholds):
                tput_demand = get_tput_demand(req_id, conf)
                logger.debug(f"conf {conf}: tput_demand={tput_demand}, ")
                logger.debug(f"  cum_tput_demand = {cum_tput_demand}")
                logger.debug(f"  cum_min_tput_suffix[{i}] = {cum_min_tput_suffix[i]}")
                logger.debug(f"  total = {cum_tput_demand + tput_demand + cum_min_tput_suffix[i]}")
                logger.debug(f"  throughput_supply = {self.throughput_supply}")
                if cum_tput_demand + tput_demand + cum_min_tput_suffix[i] <= self.throughput_supply:
                    req_state.max_confidence_threshold_idx = idx
                    logger.debug(f"  set max confidence to {req_state.max_confidence_threshold_idx}")
                    cum_tput_demand += tput_demand
                    break
                elif idx == len(self.candidate_confidence_thresholds) - 1:
                    # last confidence, cannot meet SLO
                    logger.debug(f"  could not meet SLO with any confidence, set max confidence to {self.candidate_confidence_thresholds[-1]}")
                    req_state.max_confidence_threshold_idx = len(self.candidate_confidence_thresholds) - 1
                    cum_tput_demand += tput_demand
            # else:
            #     logger.debug(f"  could not meet SLO with any confidence, set max confidence to {self.candidate_confidence_thresholds[-1]}")
            #     req_state.max_confidence_threshold_idx = len(self.candidate_confidence_thresholds) - 1
            
    def get_request_max_confidence(self, req_id: str) -> float:
        def get_tput_demand(r_id: str, confidence: float) -> float:
            num_recompute_steps_left, num_cache_steps_left = self.get_num_steps_left_by_type(r_id, confidence)
            tput_demand = (num_recompute_steps_left * self.requests[r_id].num_tokens + \
                num_cache_steps_left * self.requests[r_id].denoise_block_size) / self.requests[r_id].slo_time_remaining
            tput_demand = min(tput_demand, self.max_throughput_demand_per_req)
            return tput_demand
            
        remaining_tput_supply = self.throughput_supply
        for r_id in self.request_states:
            if r_id == req_id:
                continue
            req_state = self.request_states[r_id]
            if req_state.is_best_effort or not self.is_possible_to_meet_slo(r_id):
                continue
            assert req.slo_time_remaining > 0

            req = self.requests[r_id]
            if req.unmask_progress > self.requests[req_id].unmask_progress:
                # respect confidence
                if req_state.confidence_threshold is not None:
                    conf = req_state.confidence_threshold
                else:
                    conf = self.candidate_confidence_thresholds[0]
            else:
                conf = self.candidate_confidence_thresholds[-1]
            
            remaining_tput_supply -= get_tput_demand(r_id, conf)
        
        if remaining_tput_supply <= 0:
            return min(self.candidate_confidence_thresholds)
        
        for conf in self.candidate_confidence_thresholds:
            tput_demand = get_tput_demand(req_id, conf)
            if tput_demand <= remaining_tput_supply:
                return conf
        
        return min(self.candidate_confidence_thresholds)
        

    async def schedule(self) -> SchedulerOutput:
        # utils
        def determine_new_exec_tokens(request: Request) -> int:
            request.exec_start_pos = 0
            request.num_exec_tokens = len(request._all_token_ids)
            return request.num_exec_tokens

        def determine_running_exec_tokens(request: Request) -> int:
            request.exec_start_pos = request.cur_block_start if \
                (self.cache_prefix and not request.is_start_of_new_block) \
                    else 0
            request.num_exec_tokens = request.denoise_block_size if \
                (self.cache_suffix and not request.is_start_of_new_block) \
                    else len(request._all_token_ids) - request.exec_start_pos
            logger.debug(f"determining running exec tokens for request {request.request_id}: "
                         f"exec_start_pos={request.exec_start_pos}, "
                         f"num_exec_tokens={request.num_exec_tokens}"
                         f"is_start_of_new_block={request.is_start_of_new_block}")
            
            return request.num_exec_tokens
        
        def can_migrate(request: Request) -> bool:
            if not self.cache_prefix and not self.cache_suffix:
                return True
            
            if request.is_start_of_new_block:
                return True
            
            return False
        
        
        def add_new_request(request: Request, executor_id: int) -> None:
            scheduled_new_reqs[executor_id].append(request)
            if self.cache_prefix or self.cache_suffix:
                new_blocks = self.kv_cache_managers[executor_id].allocate_slots(
                    request,
                    request.num_tokens,
                )
                assert new_blocks is not None
                logger.debug(f"request {request.request_id} allocated new blocks {new_blocks} on executor {executor_id}")

            req_to_new_block_ids[executor_id][request.request_id] = \
                self.kv_cache_managers[executor_id].get_block_ids(
                    request.request_id)
            num_scheduled_tokens[executor_id][request.request_id] = \
                determine_new_exec_tokens(request)

            request.status = RequestStatus.RUNNING
            
        def add_running_request(request: Request, executor_id: int) -> None:
            scheduled_running_reqs[executor_id].append(request)
            req_to_new_block_ids[executor_id][request.request_id] = ()
            num_scheduled_tokens[executor_id][request.request_id] = \
                determine_running_exec_tokens(request)

            request.status = RequestStatus.RUNNING
        
        def remove_request(request: Request, executor_id: int) -> None:
            if executor_id in num_scheduled_tokens and \
               request.request_id in num_scheduled_tokens[executor_id]:
                num_scheduled_tokens[executor_id].pop(request.request_id)
                
                if len(num_scheduled_tokens[executor_id]) == 0:
                    num_scheduled_tokens.pop(executor_id)   
            
            if executor_id in req_to_new_block_ids and \
               request.request_id in req_to_new_block_ids[executor_id]:
                req_to_new_block_ids[executor_id].pop(request.request_id)
                
                if len(req_to_new_block_ids[executor_id]) == 0:
                    req_to_new_block_ids.pop(executor_id)
            
            self._free_request_on_executor(request, executor_id, False, False)
            logger.debug(f"Request {request.request_id} removed from executor {executor_id} and freed.")

            request.status = RequestStatus.WAITING
            
        
        def get_normal_batch_size(executor_id: int) -> int:
            batch_size = sum(
                self.requests[req_id].num_tokens
                for req_id in self.executor_states[executor_id].req_ids
                if self.request_states[req_id].is_normal)
            logger.debug(f"Executor {executor_id} urgent batch size: {batch_size}")
            return batch_size
        
        def get_non_urgent_batch_size(executor_id: int) -> int:
            batch_size = sum(
                self.requests[req_id].num_tokens
                for req_id in self.executor_states[executor_id].req_ids
                if not self.request_states[req_id].is_urgent)
            logger.debug(f"Executor {executor_id} non-urgent batch size: {batch_size}")
            return batch_size
        
        # def all_slos_met(req_ids: Iterable[str], tp_degree: int, batch_size: int, assigned_confidence: Optional[dict[str, float]] = {}) -> bool:
        #     for req_id in req_ids:
        #         if req_id not in assigned_confidence:
        #             conf = self.request_states[req_id].confidence_threshold 
        #             # logger.debug(f"Request {req_id} using existing confidence threshold {conf}")
        #         else:
        #             conf = assigned_confidence[req_id]
        #             # logger.debug(f"Request {req_id} using assigned confidence threshold {conf}")
        #         est_time_left = self.get_estimated_time_left(req_id, tp_degree, batch_size, conf)
        #         slo_time_remaining = self.requests[req_id].slo_time_remaining
        #         if est_time_left > slo_time_remaining:
        #             logger.debug(f"Request {req_id} SLO not met: estimated time left {est_time_left} > SLO time remaining {slo_time_remaining} ")
        #             return False
        #         else:
        #             # opportunistic and best_effort requests do not have SLOs
        #             continue
        #     return True

        def all_slos_met(target_req_ids: Iterable[str], 
                         tp_degree: int, 
                         all_req_ids: Iterable[str],
                         assigned_confidence: Optional[dict[str, float]] = {}
                         ) -> bool:
            start_time = time.time()
            for req_id in target_req_ids:
                assert req_id in all_req_ids
                if req_id not in assigned_confidence:
                    conf = self.request_states[req_id].confidence_threshold 
                    # logger.debug(f"Request {req_id} using existing confidence threshold {conf}")
                else:
                    conf = assigned_confidence[req_id]
                    # logger.debug(f"Request {req_id} using assigned confidence threshold {conf}")
                est_time_left = self.get_estimated_time_left(req_id, tp_degree, all_req_ids, conf)
                slo_time_remaining = self.requests[req_id].slo_time_remaining
                if est_time_left > slo_time_remaining:
                    logger.debug(f"Request {req_id} SLO not met: estimated time left {est_time_left} > SLO time remaining {slo_time_remaining} ")
                    logger.debug(f"all_slos_met took {time.time() - start_time} seconds")
                    return False
            logger.debug(f"all_slos_met took {time.time() - start_time} seconds")
            return True
        
        def order_by_unmask_progress(req_ids: Iterable[str], decreasing=True) -> list[str]:
            # decreasing
            ordered_reqs = sorted(
                req_ids,
                key=lambda r_id: self.requests[r_id].unmask_progress,
                reverse=decreasing
            )
            logger.debug(f"Ordered requests by unmask progress: {ordered_reqs}")
            logger.debug(f"with unmask progresses: {[self.requests[r_id].unmask_progress for r_id in ordered_reqs]}")
            return ordered_reqs
        
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        with self.schedule_profiler.section("total"):
            with self.schedule_profiler.section("update executors"):
                # update executor states
                self.update_executors()

            with self.schedule_profiler.section("copy"):
                old_request_states = copy.deepcopy(self.request_states)

            with self.schedule_profiler.section("init"):
                logger.debug(f"start of schedule, executors manager: {self.executors_manager}")
                idle_executors = []
                idle_not_accepting_executors = []

                for executor_id in self.executors_manager.executors:
                    if self.executors_manager.executors[executor_id].is_idle():
                        logger.debug(f"executor {executor_id} is idle. Status transition: IDLE -> CONSIDERED_FOR_SCHEDULING")
                        self.executors_manager.executors[executor_id].set_considered_for_scheduling()
                        idle_executors.append(executor_id)
                    elif self.executors_manager.executors[executor_id].is_idle_not_accepting_new_requests():
                        logger.debug(f"executor {executor_id} is idle_not_accepting_new_requests. Status transition: IDLE_NOT_ACCEPTING_NEW_REQUESTS -> CONSIDERED_FOR_SCHEDULING")
                        self.executors_manager.executors[executor_id].set_considered_for_scheduling()
                        idle_not_accepting_executors.append(executor_id)
                    else:
                        logger.debug(f"executor {executor_id} is not idle. Skip.")

                # idle_executors are guranteed not killed from now on
                logger.debug(f"idle executors: {idle_executors}")
                logger.debug(f"idle not accepting executors: {idle_not_accepting_executors}")
                # assert len(idle_executors) > 0

                scheduled_new_reqs: dict[int, list[Request]] = defaultdict(list)
                scheduled_resumed_reqs: dict[int, list[Request]] = defaultdict(list)
                scheduled_running_reqs: dict[int, list[Request]] = defaultdict(list)
                # this guy is not used for now
                preempted_reqs: dict[int, list[Request]] = {}

                req_to_new_block_ids: dict[int, dict[str, tuple[list[int], ...]]] = \
                    defaultdict(dict) 
                num_scheduled_tokens: dict[int, dict[str, int]] = defaultdict(dict)

                # requests_changed = set()
                requests_in_scheduler_output = set()

            with self.schedule_profiler.section("stage 0"):
                # Deal with all pending changes
                logger.debug(f"Start pending remove changes")
                for executor_id in idle_executors + idle_not_accepting_executors:
                    # note that req_ids_to_remove changes sizes during execution
                    for req_id in list(self.executor_states[executor_id].pending_request_removals):
                        logger.debug(f"Removing pending request {req_id} from its executor {executor_id}")
                        assert self.request_states[req_id].is_running
                        assert self.request_states[req_id].executor_id == executor_id
                        assert self.request_states[req_id].pending_executor_id is None
                        assert req_id in self.executor_states[executor_id].req_ids
                        
                        self.executor_states[executor_id].commit_pending_request_removal(req_id)
                        # self.request_states[req_id].commit_pending_executor()
                        self.request_states[req_id].remove_executor()
                        # requests_changed.add(req_id)
                        requests_in_scheduler_output.add(req_id)

                    logger.debug("\n")
                            
                logger.debug(f"Start pending add changes")
                for executor_id in idle_executors + idle_not_accepting_executors:
                    # note that pending_request_additions changes sizes during execution
                    for req_id in list(self.executor_states[executor_id].pending_request_additions):
                        assert self.request_states[req_id].is_scheduled or \
                            self.request_states[req_id].is_running
                        if self.request_states[req_id].is_scheduled:
                            # the request is done being removed.
                            logger.debug(f"Adding pending request {req_id} to its executor {executor_id}")
                            self.executor_states[executor_id].commit_pending_request_addition(req_id)
                            self.request_states[req_id].commit_pending_executor()
                            # requests_changed.add(req_id)
                            requests_in_scheduler_output.add(req_id)
                        elif self.request_states[req_id].is_running:
                            3/0 # should not happen
                            logger.debug(f"Request {req_id} still on executor {self.request_states[req_id].executor_id}. Cannot add to executor {executor_id} yet.")
                            # remains in running_snapshot
                        else:
                            raise Exception(f"Request {req_id} shouldn't be in unscheduled_snapshot.")
                    logger.debug("\n")
                
                # all pending changes should be done for the idle executors
                assert all(
                    self.executor_states[executor_id].has_no_pending_changes()
                    for executor_id in idle_executors + idle_not_accepting_executors
                )
                logger.debug(f"End pending changes\n")
            
            with self.schedule_profiler.section("stage 1"):
                # A (keep request on executor if possible)
                logger.debug(f"Start A (keep request on executor if possible)")
                start_time = time.time()
                for executor_id in idle_executors + idle_not_accepting_executors:
                    logger.debug(f"Processing executor {executor_id}")
                    # loop from newest to oldest requests
                    logger.debug(f"Checking running requests on executor {executor_id} for SLO compliance.")
                    # for req_id in reversed(self.executor_states[executor_id].sorted_req_ids(self.request_states)):
                    # kick out earlier progress first
                    for req_id in order_by_unmask_progress(self.executor_states[executor_id].req_ids, decreasing=False):
                        logger.debug(f"Checking request {req_id} on executor {executor_id}.")
                        max_conf_idx = self.request_states[req_id].max_confidence_threshold_idx
                        logger.debug(f"max_conf_idx: {max_conf_idx}")

                        # first update conf to min(old_conf, max_conf)
                        if self.request_states[req_id].confidence_threshold > self.candidate_confidence_thresholds[max_conf_idx]:
                            old_conf = self.request_states[req_id].confidence_threshold
                            self.request_states[req_id].confidence_threshold = self.candidate_confidence_thresholds[max_conf_idx]
                            logger.debug(f"Lowered request {req_id} confidence threshold from {old_conf} to max allowed {self.request_states[req_id].confidence_threshold}")

                        if not can_migrate(self.requests[req_id]):
                            logger.debug(f"Request {req_id} cannot migrate since not at block boundary.")
                            if self.request_states[req_id].is_best_effort or not self.is_possible_to_meet_slo(req_id):
                                logger.debug(f"best effort or can't meet slo. set confidence to highest.")
                                self.request_states[req_id].confidence_threshold = self.best_effort_confidence_threshold
                                requests_in_scheduler_output.add(req_id)
                                continue
                            # can lower confidence threshold, but cannot move to another executor
                            old_conf = self.request_states[req_id].confidence_threshold
                            # for conf in [x for x in self.candidate_confidence_thresholds]:
                            for conf in self.candidate_confidence_thresholds[max_conf_idx:]:
                                logger.debug(f"Considering confidence threshold {conf} for request {req_id}")
                                self.request_states[req_id].confidence_threshold = conf
                                if all_slos_met(
                                    [req_id],
                                    self.executor_states[executor_id].tp_degree,
                                    # batch_size,
                                    self.executor_states[executor_id].get_projected_req_ids(),
                                    {req_id: conf}):
                                    logger.debug(f"Keeping request {req_id} on executor {executor_id} with conf {conf}")
                                    # no need to add_request, bc it's already on it
                                    # do need to add it to requests_changed to make scheduled_running_reqs
                                    # requests_changed.add(req_id)
                                    break
                            if old_conf != self.request_states[req_id].confidence_threshold:
                                logger.debug(f"Updated confidence: {old_conf} ==> {self.request_states[req_id].confidence_threshold}")
                            requests_in_scheduler_output.add(req_id)
                            continue
                        else:
                            if executor_id in idle_not_accepting_executors:
                                logger.debug(f"Executor {executor_id} not accepting new requests. Removing request {req_id}.")
                                self.executor_states[executor_id].remove_request(req_id)
                                self.request_states[req_id].remove_executor()
                                self.request_states[req_id].remove_pending_executor()
                                requests_in_scheduler_output.add(req_id)
                                continue
                            # check slo
                            assert self.request_states[req_id].is_running
                            if self.request_states[req_id].is_best_effort:
                                logger.debug(f"Request {req_id} is best-effort, so skip checking SLO.")
                                requests_in_scheduler_output.add(req_id)
                                continue
                            # batch_size = self.executor_states[executor_id].get_projected_batch_size()
                            # batch_size = self.executor_states[executor_id].get_batch_size() # don't include pending changes, bc they are already committed above
                            # logger.debug(f"Checking request {req_id}. Batch size: {batch_size}")

                            flag_scheduled = False
                            # for conf in [x for x in self.candidate_confidence_thresholds if x >= self.request_states[req_id].confidence_threshold]:
                            # try to see if we can up the confidence threshold 
                            for conf in self.candidate_confidence_thresholds[max_conf_idx:]:
                                if conf < self.request_states[req_id].confidence_threshold:
                                    break
                                logger.debug(f"Considering confidence threshold {conf} for request {req_id}")
                                if not all_slos_met(
                                    [req_id],
                                    self.executor_states[executor_id].tp_degree,
                                    # batch_size,
                                    self.executor_states[executor_id].get_projected_req_ids(),
                                    {req_id: conf}):
                                    continue

                                logger.debug(f"Keeping request {req_id} on executor {executor_id} with conf {conf}")
                                # no need to add_request, bc it's already on it
                                # do need to add it to requests_changed to make scheduled_running_reqs
                                # requests_changed.add(req_id)
                                requests_in_scheduler_output.add(req_id)
                                self.request_states[req_id].confidence_threshold = conf
                                # priority remains the same
                                flag_scheduled = True
                                break
                            if not flag_scheduled:
                                logger.debug(f"Removing request {req_id} from executor {executor_id} due to SLO miss")
                                # remove_request(request, executor_id)
                                # cluster_snapshot[executor_id].pop(req_id)
                                self.executor_states[executor_id].remove_request(req_id)
                                self.request_states[req_id].remove_executor()
                                self.request_states[req_id].remove_pending_executor()
                                # requests_changed.add(req_id)
                                requests_in_scheduler_output.add(req_id)
                                
                        logger.debug("\n")
                logger.debug(f"End A\n")
                logger.debug(f"A took {time.time() - start_time} seconds")
                # finished A for all executors

            with self.schedule_profiler.section("stage 2"):
                # B
                start_time = time.time()
                logger.debug(f"Start B (schedule unscheduled normal & opportunistic requests)")
                # unscheduled_requests = deque(order_by_unmask_progress(
                #     [r_id for r_id in self.request_states if self.request_states[r_id].is_unscheduled and \
                #         self.is_possible_to_meet_slo(r_id)]))
                unscheduled_requests = [
                    (-self.requests[r_id].unmask_progress, r_id) for r_id in self.request_states if self.request_states[r_id].is_unscheduled and \
                    self.is_possible_to_meet_slo(r_id)]
                heapq.heapify(unscheduled_requests)
                
                while len(unscheduled_requests) > 0:
                    # maybe pick the request with the most progress?
                    # req_id = unscheduled_requests.popleft()
                    req_id = heapq.heappop(unscheduled_requests)[1]
                    request = self.requests[req_id]
                    logger.debug(f"Processing waiting unscheduled request {req_id}.")
                    
                    # loop through executors, in (ascending TP_degree, descending normal batch size, ascending total batch size) order
                    flag_scheduled = False
                    # for conf in self.candidate_confidence_thresholds:
                    max_conf_idx = self.request_states[req_id].max_confidence_threshold_idx
                    logger.debug(f"max_conf_idx for request {req_id}: {max_conf_idx}")
                    for conf in self.candidate_confidence_thresholds[max_conf_idx:]:
                        if flag_scheduled:
                            break
                        logger.debug(f"Trying confidence threshold {conf}.")
                        for ex_id in sorted(
                            [e_id for e_id in self.executor_states if not self.executors_manager.executors[e_id].is_killing() and not self.executors_manager.waiting_to_be_killed[e_id] and self.executor_states[e_id].has_no_pending_changes()],
                            key=lambda eid: (
                                self.executor_states[eid].tp_degree,
                                -get_normal_batch_size(eid),
                                self.executor_states[eid].get_batch_size() - get_normal_batch_size(eid)
                                )
                            ):
                            if flag_scheduled:
                                break
                            logger.debug(f"Considering executor {ex_id}.")

                            batch_size = request.num_tokens + get_normal_batch_size(ex_id)
                            if batch_size > self.max_num_scheduled_tokens:
                                logger.debug(f"Can't schedule due to batch size {batch_size} > max {self.max_num_scheduled_tokens}. Skipping executor.")
                                continue

                            # check if all normal reqs can meet SLO
                            requests_scheduled = [req_id] + self.executor_states[ex_id].get_normal_req_ids(self.request_states)

                            if not all_slos_met(
                                requests_scheduled,
                                self.executor_states[ex_id].tp_degree,
                                # batch_size,
                                requests_scheduled,
                                {req_id: conf}):
                                logger.debug(f"Can't schedule due to SLO miss when considering all normal requests. Skipping executor.")
                                continue
                            
                            # scheduling this request to this executor
                            if ex_id in idle_executors:
                                # add now
                                logger.debug(f"Executor {ex_id} is idle, so adding urgent request {req_id} to it with confidence {conf}.")
                                self.executor_states[ex_id].add_request(request)
                                self.request_states[req_id].set_executor(ex_id, RequestStatePriority.NORMAL, conf)
                                self.request_states[req_id].set_pending_executor(ex_id, RequestStatePriority.NORMAL, conf)
                                requests_in_scheduler_output.add(req_id)
                            else:
                                # add later
                                logger.debug(f"Executor {ex_id} is not idle, so marking urgent request {req_id} for pending addition with confidence {conf}.")
                                self.executor_states[ex_id].mark_pending_request_addition(request)
                                self.request_states[req_id].set_pending_executor(ex_id, RequestStatePriority.NORMAL, conf)
                            
                            # requests_changed.add(req_id)
                            flag_scheduled = True

                            # look at opportunistic then best-effort requests, kicking off if necessary
                            for r_id in self.executor_states[ex_id].get_opportunistic_req_ids(self.request_states) + \
                                self.executor_states[ex_id].get_best_effort_req_ids(self.request_states):
                                assert self.request_states[r_id].is_running
                                logger.debug(f"looking at request {r_id} on executor {ex_id} to see if it can be kept.")
                                logger.debug(f"self.executor_states[{ex_id}]: {self.executor_states[ex_id]}")

                                remove_request_flag = False
                                
                                new_batch_size = batch_size + self.requests[r_id].num_tokens
                                batch_size_exceeded = new_batch_size > self.max_num_scheduled_tokens
                                remove_request_flag |= batch_size_exceeded
                                logger.debug(f"batch size exceeded: {batch_size_exceeded}")

                                if not remove_request_flag:
                                    slo_miss = not all_slos_met(
                                        [r for r in (requests_scheduled + [r_id]) if not self.request_states[r].is_best_effort],
                                        self.executor_states[ex_id].tp_degree,
                                        # new_batch_size,
                                        requests_scheduled + [r_id],
                                        {req_id: conf})
                                    logger.debug(f"slo miss: {slo_miss}")
                                    remove_request_flag |= slo_miss

                                if remove_request_flag:
                                    logger.debug(f"Removing {"opportunistic" if self.request_states[r_id].is_opportunistic else "best-effort"} request {r_id}")
                                    
                                    if ex_id in idle_executors:
                                        logger.debug(f"Executor {ex_id} is idle, so removing request {r_id} immediately.")
                                        self.executor_states[ex_id].remove_request(r_id)
                                        self.request_states[r_id].remove_executor()
                                        self.request_states[r_id].remove_pending_executor()
                                        requests_in_scheduler_output.add(req_id)

                                        # if self.request_states[r_id].is_unscheduled:
                                        #     unscheduled_requests.add(r_id)
                                        assert self.request_states[r_id].is_unscheduled
                                        if self.is_possible_to_meet_slo(r_id):
                                            # unscheduled_requests.add(r_id)
                                            unscheduled_requests.append((-self.requests[r_id].unmask_progress, r_id))
                                            heapq.heapify(unscheduled_requests)
                                    else:
                                        logger.debug(f"Executor {ex_id} is not idle, so marking request {r_id} for pending removal.")
                                        self.executor_states[ex_id].mark_pending_request_removal(r_id)
                                        self.request_states[r_id].remove_pending_executor()
                                    
                                else:
                                    requests_scheduled.append(r_id)
                                    batch_size += self.requests[r_id].num_tokens
                                    logger.debug(f"Keeping opportunistic request {r_id} on executor {ex_id}. Updated batch size: {batch_size}")
                                        

                    if not flag_scheduled:
                        logger.debug(f"Cannot schedule urgent request {req_id} to any idle executor due to SLO miss. Skipping for now.")
                        # if not possible (likely due to max_num_batch_tokens), skip for now
                        # skipped_waiting_requests.add_request(request)
                        # skipped_waiting.append(req_id)
                        # just find the lowest load executor and assign
                        # no need to set request_states_snapshot because nothing has changed
                
                logger.debug(f"End B\n")
                logger.debug(f"B took {time.time() - start_time} seconds")
            
            with self.schedule_profiler.section("stage 3"):
                start_time = time.time()
                # logger.debug(f"Start C (schedule opportunistic requests)")
                # Opportunistic promoting (only looking at requests on idle executors)
                # only promote to executors with no requests scheduled at all
                promoted_req_ids = set()
                
                zero_load_executors = [
                    ex_id
                    for ex_id in idle_executors
                    if self.executor_states[ex_id].is_drained()
                ]
                while len(zero_load_executors) > 0:
                    # find the executor with highest TP degree
                    tgt_ex_id = max(zero_load_executors, key=lambda eid: self.executor_states[eid].tp_degree)
                    logger.debug(f"Looking at zero load executor {tgt_ex_id} with TP degree {self.executor_states[tgt_ex_id].tp_degree}")
                    logger.debug(f"{self.executor_states[tgt_ex_id]}")
                    # find a job to promote
                    # find the job with the least (priority, conf, slo_time_remaining - est_time_left)
                    best_req_id = None
                    best_metric = (RequestStatePriority.BEST_EFFORT + 1, 1.1, float('inf'))
                    # only look at req on idle executors
                    for src_ex_id in idle_executors:
                        for req_id in self.executor_states[src_ex_id].req_ids:
                            if req_id in promoted_req_ids:
                                continue
                            if not can_migrate(self.requests[req_id]):
                                logger.debug(f"Request {req_id} cannot migrate since not at block boundary.")
                                requests_in_scheduler_output.add(req_id)
                                continue
                            logger.debug(f"Considering request {req_id} on executor {src_ex_id} for promotion to executor {tgt_ex_id}")
                            num_tokens = self.requests[req_id].num_tokens
                            _, src_step_latency = self.get_profile_latency(
                                self.executor_states[src_ex_id].tp_degree, self.executor_states[src_ex_id].get_projected_batch_size())
                            _, tgt_step_latency = self.get_profile_latency(
                                self.executor_states[tgt_ex_id].tp_degree, num_tokens)
                            logger.debug(f"src_step_latency: {src_step_latency}")
                            logger.debug(f"tgt_step_latency: {tgt_step_latency}")
                            if tgt_step_latency < src_step_latency:
                                # possibly just pick the first one we see that satisfies this 
                                est_time_left = self.get_estimated_time_left(
                                    req_id, 
                                    self.executor_states[src_ex_id].tp_degree,
                                    # self.executor_states[src_ex_id].get_batch_size(),
                                    self.executor_states[src_ex_id].req_ids,
                                    self.request_states[req_id].confidence_threshold)
                                slo_time_remaining = self.requests[req_id].slo_time_remaining
                                metric = (
                                    self.request_states[req_id].priority,
                                    self.request_states[req_id].confidence_threshold,
                                    # est_time_left - slo_time_remaining
                                    slo_time_remaining - est_time_left
                                )
                                if metric < best_metric:
                                    best_metric = metric
                                    best_req_id = req_id
                                    logger.debug(f"Found a better candidate for promotion: request {req_id} from executor {src_ex_id} with metric {metric}")
                    
                    logger.debug(f"Best request to promote to executor {tgt_ex_id}: {best_req_id}")
                    # promote it
                    if best_req_id is not None:
                        src_ex_id = self.request_states[best_req_id].executor_id
                        logger.debug(f"Promoting opportunistic request {best_req_id} from executor {src_ex_id} to executor {tgt_ex_id}")
                        assert src_ex_id != tgt_ex_id

                        # find the highest confidence that can meet SLO on the new executor
                        # if best effort, just use the highest confidence
                        if self.request_states[best_req_id].is_best_effort:
                            new_conf = self.best_effort_confidence_threshold
                        else:
                            new_conf = None
                            # for conf in self.candidate_confidenkce_thresholds:
                            max_conf_idx = self.request_states[best_req_id].max_confidence_threshold_idx
                            logger.debug(f"max_conf_idx for request {best_req_id}: {max_conf_idx}")
                            for conf in self.candidate_confidence_thresholds[max_conf_idx:]:
                                if all_slos_met(
                                    [best_req_id],
                                    self.executor_states[tgt_ex_id].tp_degree,
                                    # self.requests[best_req_id].num_tokens,
                                    [best_req_id],
                                    {best_req_id: conf}):
                                    new_conf = conf
                                    break
                        logger.debug(f"new_conf: {new_conf}. Old conf: {self.request_states[best_req_id].confidence_threshold}")
                        # assert new_conf is not None
                        # super rare case: request becomes best-effort DURING promotion
                        # we just use the lowest confidence then
                        if new_conf is None:
                            new_conf = min(self.candidate_confidence_thresholds)
                        # assert new_conf >= self.request_states[best_req_id].confidence_threshold
                        # if src_time_left and tgt_time_left are very close, the confidence might actually drop
                        # it's fine...

                        # remove from src
                        self.executor_states[src_ex_id].remove_request(best_req_id)
                        self.request_states[best_req_id].remove_executor()
                        self.request_states[best_req_id].remove_pending_executor()
                        
                        priority = RequestStatePriority.OPPORTUNISTIC if not self.request_states[best_req_id].is_best_effort else RequestStatePriority.BEST_EFFORT
                        self.executor_states[tgt_ex_id].add_request(self.requests[best_req_id])
                        self.request_states[best_req_id].set_executor(tgt_ex_id, priority, new_conf)
                        self.request_states[best_req_id].set_pending_executor(tgt_ex_id, priority, new_conf)
                        
                        promoted_req_ids.add(best_req_id)
                        requests_in_scheduler_output.add(best_req_id)

                        zero_load_executors.remove(tgt_ex_id)
                        if len(self.executor_states[src_ex_id].req_ids) == 0:
                            zero_load_executors.append(src_ex_id)
                    else:
                        # nothing to promote
                        logger.debug(f"Nothing to promote to executor {tgt_ex_id}.")
                        break
                
                logger.debug(f"End C\n")
                logger.debug(f"C took {time.time() - start_time} seconds")

            with self.schedule_profiler.section("stage 4"):
                start_time = time.time()

                logger.debug(f"Start D (schedule best-effort requests)")
                # Best Effort
                # These requests has estimated time left > SLO remaining time even at the lowest confidence threshold,
                # given current cluster load.
                # Schedule them onto whichever executor with enough token budget and
                # the existing requests on it can still meet SLO.
                for req_id in [r_id for r_id in self.request_states if self.request_states[r_id].is_unscheduled]:
                    logger.debug(f"Processing best-effort request {req_id}.")
                    conf = self.best_effort_confidence_threshold
                    # conf = min(self.candidate_confidence_thresholds)
                    for executor_id in idle_executors:
                        logger.debug(f"Considering executor {executor_id}")
                        # for idle executors, still possible there are pending add
                        new_batch_size = self.requests[req_id].num_tokens + self.executor_states[executor_id].get_batch_size()
                        
                        if new_batch_size > self.max_num_scheduled_tokens:
                            logger.debug(f"Can't schedule due to batch size {new_batch_size} > max {self.max_num_scheduled_tokens}. Skipping executor.")
                            continue

                        normal_and_opportunistic_req_ids = self.executor_states[executor_id].get_conditional_req_ids(
                            self.request_states, {RequestStatePriority.NORMAL, RequestStatePriority.OPPORTUNISTIC})
                        if not all_slos_met(
                            normal_and_opportunistic_req_ids,
                            self.executor_states[executor_id].tp_degree,
                            normal_and_opportunistic_req_ids + [req_id],
                            {req_id: conf}):
                            logger.debug(f"Can't schedule due to SLO miss when considering normal and opportunistic requests. Skipping executor.")
                            continue
                        
                        # schedule!
                        self.executor_states[executor_id].add_request(self.requests[req_id])
                        self.request_states[req_id].set_executor(executor_id, RequestStatePriority.BEST_EFFORT, conf)
                        self.request_states[req_id].set_pending_executor(executor_id, RequestStatePriority.BEST_EFFORT, conf)
                        logger.debug(f"Scheduling best-effort request {req_id} to executor {executor_id} with confidence {conf}.")

                        requests_in_scheduler_output.add(req_id)
                        break
                
                logger.debug(f"End D\n")
                logger.debug(f"D took {time.time() - start_time} seconds")
            
            with self.schedule_profiler.section("stage 5"):
                start_time = time.time()
                # For the remaining unscheduled, we just skip...

                unscheduled_requests = set(
                    [r_id for r_id in self.request_states if self.request_states[r_id].is_unscheduled]
                )
                logger.debug(f"Unscheduled requests after scheduling: {unscheduled_requests}")


                # Actually add and remove requests
                logger.debug(f"Finalizing scheduling decisions and updating states.")
                for req_id in requests_in_scheduler_output:
                    if old_request_states[req_id].executor_id != self.request_states[req_id].executor_id:
                        logger.debug(f"Request {req_id} changed executor from {old_request_states[req_id].executor_id} to {self.request_states[req_id].executor_id}.")
                        # None -> Some
                        # Some -> None
                        # Some -> Some (different)
                        
                        # remove from old
                        if old_request_states[req_id].executor_id is not None:
                            remove_request(
                                self.requests[req_id], old_request_states[req_id].executor_id)
                        if self.request_states[req_id].executor_id is not None:
                            add_new_request(
                                self.requests[req_id], self.request_states[req_id].executor_id)
                    else:
                        # if it's a future change, don't do anything
                        logger.debug(f"Request {req_id} remains on the same executor {self.request_states[req_id].executor_id}.")
                        if self.request_states[req_id].executor_id is None:
                            # None -> None
                            # do nothing
                            # this is possible if a request has pending addition, but 
                            # then got removed later due to SLO miss
                            continue

                        assert self.request_states[req_id].executor_id in idle_executors + idle_not_accepting_executors
                        # if self.request_states[req_id].executor_id not in idle_executors:
                        #     # Some -> Some (same, but pending change)
                        #     # do nothing for now
                        #     pass
                        # if old_request_states[req_id].pending_executor_id not in idle_executors:
                        #     pass
                        # if self.request_states[req_id].executor_id is None:
                        #     # None -> None
                        #     # do nothing
                        #     pass
                        add_running_request(
                            self.requests[req_id], self.request_states[req_id].executor_id)
                        
                logger.debug(f"E took {time.time() - start_time} seconds")

            with self.schedule_profiler.section("stage 6"):
                
                start_time = time.time()
                    
                # Construct the scheduler output.
                scheduler_outputs: dict[int, SchedulerOutput] = {}

                for executor_id in idle_executors + idle_not_accepting_executors:
                    if executor_id not in num_scheduled_tokens and \
                        not self.free_req_ids[executor_id]:
                        logger.debug(f"Executor {executor_id} has no requests scheduled to run, and no finished requests. skip")
                        # not scheduled to run at all, and no finished reqs. skip
                        # executor should not be in idle_not_accepting_executors
                        assert executor_id not in idle_not_accepting_executors
                        async with self.executors_manager.cond[executor_id]:
                            logger.debug(f"Executor {executor_id} status transition: CONSIDERED_FOR_SCHEDULING -> IDLE")
                            self.executors_manager.executors[executor_id].set_idle()
                            self.executors_manager.cond[executor_id].notify()
                        continue

                    logger.debug(f"Executor {executor_id} status transition: CONSIDERED_FOR_SCHEDULING -> SCHEDULED")
                    self.executors_manager.executors[executor_id].set_scheduled()
                    logger.debug(f"req_to_new_block_ids: {req_to_new_block_ids[executor_id]}")
                    
                    new_reqs_data = [
                        NewRequestData.from_request(req, 
                                    req_to_new_block_ids[executor_id][req.request_id])
                        for req in scheduled_new_reqs[executor_id]
                    ]
                    cached_reqs_data = self._make_cached_request_data(
                        scheduled_running_reqs[executor_id],
                        scheduled_resumed_reqs[executor_id],
                        num_scheduled_tokens[executor_id],
                        {},
                        req_to_new_block_ids[executor_id],
                    )
                    # TODO: decide confidence thresholds
                    confidence_thresholds = {}
                    for req_id in num_scheduled_tokens[executor_id].keys():
                        req_confidence_threshold = self.requests[req_id].sampling_params.confidence_threshold
                        if req_confidence_threshold is not None:
                            confidence_thresholds[req_id] = req_confidence_threshold
                            logger.debug(f"using user-defined confidence threshold {confidence_thresholds[req_id]} for req {req_id}")
                        else:
                            confidence_thresholds[req_id] = self.request_states[req_id].confidence_threshold
                            logger.debug(f"using scheduler-defined confidence threshold {confidence_thresholds[req_id]} for req {req_id}")
                    
                    exec_start_pos: dict[str, int] = {}
                    
                    for req in new_reqs_data:
                        exec_start_pos[req.req_id] = req.exec_start_pos
                    for idx, req_id in enumerate(cached_reqs_data.req_ids):
                        exec_start_pos[req_id] = cached_reqs_data.exec_start_pos[idx]

                    scheduler_output = SchedulerOutput(
                        scheduled_new_reqs=new_reqs_data,
                        scheduled_cached_reqs=cached_reqs_data,
                        exec_start_pos=exec_start_pos,
                        num_scheduled_tokens=num_scheduled_tokens[executor_id],
                        total_num_scheduled_tokens=sum(
                            num_scheduled_tokens[executor_id].values()),
                        scheduled_spec_decode_tokens={},
                        scheduled_encoder_inputs={},
                        num_common_prefix_blocks=[],
                        # finished_req_ids is an existing state in the scheduler,
                        # instead of being newly scheduled in this step.
                        # It contains the request IDs that are finished in between
                        # the previous and the current steps.
                        finished_req_ids=self.finished_req_ids[executor_id],
                        free_req_ids=self.free_req_ids[executor_id],
                        free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
                        structured_output_request_ids={},
                        grammar_bitmask=None,
                        confidence_thresholds=confidence_thresholds
                    )
                    scheduler_outputs[executor_id] = scheduler_output

                    self._update_after_schedule(executor_id, scheduler_output)
                # self._update_after_schedule(0, scheduler_output)
                logger.debug(f"returning scheduler output: {scheduler_outputs}")

                self.system_logger.log()
                logger.debug(f"last part took {time.time() - start_time} seconds")
        
        self.schedule_profiler.add_info(f"num_executors", len(self.executor_states))
        self.schedule_profiler.add_info(f"num_idle_executors", len(idle_executors))
        self.schedule_profiler.add_info(f"num_requests", len(self.request_states))
        
        self.schedule_profiler.commit()
        
        return scheduler_outputs

    def _update_after_schedule(
        self,
        executor_id: int,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.set_in_execution(True)

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids[executor_id] = set()
        self.free_req_ids[executor_id] = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_block_ids: dict[str, tuple[list[int], ...]],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[tuple[int, int]]] = []
        new_block_ids: list[tuple[list[int], ...]] = []
        num_computed_tokens: list[int] = []
        num_denoise_ran: list[int] = []
        cur_block_start: list[int] = []
        denoise_block_size: list[int] = []
        exec_start_pos: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)

            logger.debug(
                f"making cached request data for req_id: {req_id}, "
                f"use_pp: {self.use_pp}, use_connector: {use_connector}"
            )
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.unmasked_token_ids[-req.num_last_unmasked_tokens:]
                new_token_ids.append(token_ids)
                logger.debug(f"using pp, new_token_ids: {new_token_ids}")
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(req_to_new_block_ids[req_id])
            num_computed_tokens.append(req.num_computed_tokens)
            num_denoise_ran.append(req.num_denoise_ran)
            cur_block_start.append(req.cur_block_start)
            denoise_block_size.append(req.denoise_block_size)
            exec_start_pos.append(req.exec_start_pos)
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_denoise_ran=num_denoise_ran,
            cur_block_start=cur_block_start,
            denoise_block_size=denoise_block_size,
            exec_start_pos=exec_start_pos,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # The encoder input is already computed and cached.
                continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if (not self.encoder_cache_manager.can_allocate(request, i)
                    or num_encoder_tokens > encoder_budget):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    async def update_from_output(
        self,
        executor_id: int,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            request.set_in_execution(False)

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []
            logger.debug(f"generated_token_ids for req {req_id}: {generated_token_ids}")

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens, where is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                       len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                max_conf_idx = self.request_states[req_id].max_confidence_threshold_idx
                logger.debug(f"in update from output max_conf_idx  {req_id}: {max_conf_idx}")
                max_conf = self.candidate_confidence_thresholds[max_conf_idx]
                stats, new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, 
                    scheduler_output.confidence_thresholds[req_id], 
                    model_runner_output.avg_output_confidences[req_index],
                    max_conf)
                # stats.confidence_threshold = \
                #     scheduler_output.confidence_thresholds[req_id]
                self.step_estimator.add_data_point(stats)

                self.request_states[req_id].last_stats = stats
                self.update_step_estimates(req_id)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                logger.debug(f"Request {req_id} is stopped.")
                kv_transfer_params = self._free_request(request, True)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            # spec_token_ids comes from the model runner output
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                if self.structured_output_manager.should_advance(request):
                    metadata = request.structured_output_request
                    # Needs to happen after new_token_ids are accepted.
                    request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                        spec_token_ids[req_index])
                else:
                    request.spec_token_ids = spec_token_ids[req_index]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        num_cached_tokens=request.num_cached_tokens,
                        num_denoise_ran=request.num_denoise_ran,
                    ))

            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors


        # Remove the stopped requests from the running and waiting queues.
        # if stopped_running_reqs:
        #     self.running = [
        #         req_id for req_id in self.running if req_id not in [r.request_id for r in stopped_running_reqs]
        #     ]
        # if stopped_preempted_reqs:
        #     # This is a rare case and unlikely to impact performance.
        #     # self.waiting.remove_requests(stopped_preempted_reqs)
        #     self.scheduled = [
        #         req_id for req_id in self.scheduled if req_id not in [r.request_id for r in stopped_preempted_reqs]
        #     ]
        #     self.unscheduled = [
        #         req_id for req_id in self.unscheduled if req_id not in [r.request_id for r in stopped_preempted_reqs]
        #     ]

        # KV Connector: update state for finished KV Transfers.
        self._update_from_kv_xfer_finished(model_runner_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        if self.finished_req_ids_dict:
            finished_req_ids = self.finished_req_ids_dict[executor_id]
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        if engine_core_outputs:
            # Return stats to only one of the front-ends.
            next(iter(engine_core_outputs.values())).scheduler_stats = (
                self.make_stats(executor_id, spec_decoding_stats))

         # now that we have updated everything, set executor to IDLE
        async with self.executors_manager.cond[executor_id]:
            logger.debug(f"In update_from_output, executor {executor_id} status transition: OUTPUT_READY -> IDLE")
            if self.cache_prefix or self.cache_suffix:
                if self.executors_manager.waiting_to_be_killed[executor_id] and not self.executor_states[executor_id].is_drained():
                    self.executors_manager.executors[executor_id].set_idle_not_accepting_new_requests()
                else:
                    self.executors_manager.executors[executor_id].set_idle()
            else:
                self.executors_manager.executors[executor_id].set_idle()
            self.executors_manager.cond[executor_id].notify()

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[tuple[int, int]],
        confidence_threshold: float,
        avg_output_confidence: float,
        max_confidence_threshold: Optional[float] = None,
    ) -> tuple[list[tuple[int, int]], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        stats = request.update_from_output(
            new_token_ids,
            confidence_threshold,
            avg_output_confidence,
            max_confidence_threshold
        )

        stopped = check_stop(request, self.max_model_len)
        # for num_new, output_token_id in enumerate(new_token_ids, 1):
        #     request.append_output_token_ids(output_token_id)

        #     # Check for stop and update request state.
        #     # This must be called before we make the EngineCoreOutput.
        #     stopped = check_stop(request, self.max_model_len)
        #     if stopped:
        #         del new_token_ids[num_new:]  # Trim new tokens if needed.
        #         break
        return stats, new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = (
            self.encoder_cache_manager.get_cached_input_ids(request))
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_positions = request.mm_positions[input_id]
            start_pos = mm_positions.offset
            num_tokens = mm_positions.length
            if start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        # return len(self.running), len(self.scheduled) + len(self.unscheduled)
        running_reqs = len([req_id for req_id in self.request_states if self.request_states[req_id].is_running])
        scheduled_reqs = len([req_id for req_id in self.request_states if self.request_states[req_id].is_scheduled])
        unscheduled_reqs = len([req_id for req_id in self.request_states if self.request_states[req_id].is_unscheduled])

        return running_reqs, scheduled_reqs + unscheduled_reqs

    def add_request(self, request: Request) -> None:
        if len(self.executor_states) == 0:
            # we need this for update_request_max_confidence_thresholds
            self.update_executors()
        logger.debug(f"Scheduler adding request {request.request_id}")
        # self.unscheduled.append(request.request_id)
        self.requests[request.request_id] = request
        self.request_states[request.request_id] = RequestState(request)
        self.update_step_estimates(request.request_id)
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)
        self.system_logger.log()

        self.update_request_max_confidence_thresholds()

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = []
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.append(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        # for request in running_requests_to_remove:
        #     # self.running.remove(request)
        #     self.running = [
        #         req_id for req_id in self.running if req_id != request.request_id]
        # if waiting_requests_to_remove:
        #     # self.waiting.remove_requests(waiting_requests_to_remove)
        #     self.scheduled = [
        #         req_id for req_id in self.scheduled if req_id not in [req.request_id for req in waiting_requests_to_remove]
        #     ]
        #     self.unscheduled = [
        #         req_id for req_id in self.unscheduled if req_id not in [req.request_id for req in waiting_requests_to_remove]
        #     ]

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request, True)
        
    def _free_request_on_executor(self, request: Request, executor_id: int, 
                      finished: bool, prune: bool = True) -> Optional[dict[str, Any]]:
        logger.debug(f"in _free_request_on_executor, request: {request.request_id}, executor_id: {executor_id}, finished: {finished}")
        request_id = request.request_id

        # # [tau_chang] Not used start
        # delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        # self.encoder_cache_manager.free(request)
        # # [tau_chang] Not used end

        self.free_req_ids[executor_id].add(request_id)
        if finished:
            assert executor_id in self.request_states[request_id].executors_to_free

            self.finished_req_ids[executor_id].add(request_id)

            if self.finished_req_ids_dict is not None:
                self.finished_req_ids_dict[executor_id][request.client_index].\
                    add(request_id)

            if executor_id == self.request_states[request_id].executor_id:
                self.request_states[request_id].remove_executor()
                self.executor_states[executor_id].remove_request(request_id)
            else:
                self.request_states[request_id].remove_executor_to_free(executor_id)

        # otherwise, triggered by schedule(), and request_states and executor_states are updated there

        # if not delay_free_blocks:
        #     self._free_blocks(request, executor_id)
        self._free_blocks(request, executor_id)

        # update states
        # if executor_id == self.request_states[request_id].executor_id:
        #     self.request_states[request_id].remove_executor()
        # else:
            # self.request_states[request_id].remove_executor_to_free(executor_id)
        # self.request_states[request_id].remove_pending_executor()

        # self.executor_states[executor_id].remove_request(request_id)
        if prune and not self.request_states[request_id].executors_to_free:
            del self.request_states[request_id]
            del self.requests[request_id]
            self.system_logger.log(write_out=True)
        
        return None
        

    def _free_request(self, request: Request, 
                      finished: bool) -> Optional[dict[str, Any]]:
        logger.debug(f"in _free_request, request: {request}, finished: {finished}")

        # assert request.request_id in self.running
        assert self.request_states[request.request_id].is_running
        # still possible that in the next iter it gets scheduled to another executor
        cur_executor_id = self.request_states[request.request_id].executor_id
        pending_executor_id = self.request_states[request.request_id].pending_executor_id
        if pending_executor_id is not None and pending_executor_id != cur_executor_id:
            self.executor_states[pending_executor_id].unmark_pending_request_addition(request.request_id)
        # if request.request_id in self.scheduled:
        #     pending_executor_id = self.request_states[request.request_id].pending_executor_id
        #     logger.debug(f"removing to_add request {request.request_id} from pending executor {pending_executor_id}")
        #     self.executor_states[pending_executor_id].remove_request_to_add(request.request_id)

        for executor_id in list(self.request_states[request.request_id].executors_to_free):
            self._free_request_on_executor(request, executor_id, finished)
        
        self.step_estimator.save_data()
        self.update_request_max_confidence_thresholds()
        return None


    def _free_blocks(self, request: Request, executor_id: int):
        # assert request.is_finished()
        # executor_id = self.request_to_executor.get(request.request_id)
        self.kv_cache_managers[executor_id].free(request)
        self.kv_cache_managers[executor_id].free_block_hashes(request)
        logger.debug(f"Freed blocks for request {request.request_id} on executor {executor_id}")
        # del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        # return len(self.scheduled) + len(self.unscheduled) + len(self.running)
        return len(self.request_states)

    def has_finished_requests(self) -> bool:
        # logger.debug(f"in has_finished_requests, finished_req_ids: {self.finished_req_ids}")
        # return any(len(finished) > 0 for finished in self.finished_req_ids.values())
        logger.debug(f"in has_finished_requests, free_req_ids: {self.free_req_ids}")
        return any(len(free) > 0 for free in self.free_req_ids.values())
        # return len(self.finished_req_ids) > 0

    def has_not_in_execution_requests(self) -> bool:
        """Returns True if there are requests that are not in execution."""
        for req in self.requests.values():
            logger.debug(f"request: {req}, is_in_execution: {req.is_in_execution}")
        logger.debug(f"in has_not_in_execution_requests, result is {any(not req.is_in_execution for req in self.requests.values())}")
        return any(not req.is_in_execution for req in self.requests.values())

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        executor_id: int,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_managers[executor_id].make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        num_running_reqs = len([req_id for req_id in self.request_states if self.request_states[req_id].is_running])
        
        return SchedulerStats(
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=len(self.request_states) - num_running_reqs,
            kv_cache_usage=self.kv_cache_managers[executor_id].usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(self.requests[req_id].is_output_corrupted
                                   for req_id in self.request_states if self.request_states[req_id].is_running),
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        self.step_estimator.save_data()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(
            self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        # Now that the blocks are ready, actually cache them.
        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        # Handle the case where num request tokens less then one block.
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        # This will cache the blocks iff caching is enabled.
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

        # Update the request state for scheduling.
        request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self,
                                      model_runner_output: ModelRunnerOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            scheduler the request during the next step.
        """
        # KV Connector:: update recv and send status from last step.
        for req_id in (model_runner_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (model_runner_output.finished_sending or ()):
            logger.debug("Finished sending KV transfer for request %s", req_id)
            self._free_blocks(self.requests[req_id])

    def add_executor(self, executor_id: int) -> None:
        assert executor_id not in self.executor_states

        executor = self.executors_manager.executors[executor_id]

        self.executor_states[executor.id] = ExecutorState(executor, self.max_num_scheduled_tokens)
        self.kv_cache_managers[executor.id] = KVCacheManager(
            kv_cache_config=self.executors_manager.\
                get_scheduler_kv_cache_config(executor.id),
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )

        self.throughput_supply += self.latency_profiles[self.executor_states[executor.id].tp_degree].max_throughput
        logger.debug(f"Throughput supply updated to {self.throughput_supply} after adding executor {executor_id}.")
        logger.debug(f"Added executor {executor.id} to scheduler.")

        # recompute max tp degree
        self.max_tp_degree = max([exec_state.tp_degree for exec_state in self.executor_states.values()])
            
    def remove_executor(self, executor_id: int) -> None:
        assert executor_id in self.executor_states
        for req_id in self.executor_states[executor_id].req_ids:
            self.request_states[req_id].remove_executor()
            self.request_states[req_id].remove_pending_executor()
        # for pending removals, just treat them like they have been removed
        for req_id in self.executor_states[executor_id].pending_request_removals:
            self.request_states[req_id].remove_executor()
        # for pending additions, set their pending_executor_id to None
        # they will be picked up as unscheduled
        for req_id in self.executor_states[executor_id].pending_request_additions:
            self.request_states[req_id].remove_pending_executor()
        # for req_id in self.executor_states[executor_id].pending_request_removals:
        #     self.request_states[req_id].remove_pending_executor()

        self.throughput_supply -= self.latency_profiles[self.executor_states[executor_id].tp_degree].max_throughput
        logger.debug(f"Throughput supply updated to {self.throughput_supply} after removing executor {executor_id}.")

        del self.executor_states[executor_id]
        del self.kv_cache_managers[executor_id]
        # for the rare case that this executor has never had any requests completed
        # we use if
        if executor_id in self.free_req_ids:
            del self.free_req_ids[executor_id]
        
        logger.debug(f"Removed executor {executor_id} from scheduler.")
        # recompute max tp degree
        self.max_tp_degree = max([exec_state.tp_degree for exec_state in self.executor_states.values()])

    def update_executors(self) -> None:
        for executor_id in self.executors_manager.executors:
            if executor_id not in self.executor_states:
                self.add_executor(executor_id)
        
        for executor_id in list(self.executor_states.keys()):
            if executor_id not in self.executors_manager.executors or \
                self.executors_manager.executors[executor_id].is_killing():
                self.remove_executor(executor_id)

        self.max_throughput_demand_per_req = max(
            [self.latency_profiles[exec_state.tp_degree].max_throughput
             for exec_state in self.executor_states.values()] or [0]
        )