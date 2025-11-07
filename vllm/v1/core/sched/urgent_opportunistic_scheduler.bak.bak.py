# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import bisect
from collections import defaultdict, OrderedDict
from collections.abc import Iterable
import itertools
import json
import os
import time
from typing import Any, Optional, Union

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
from vllm.v1.core.sched.utils import check_stop, LatencyProfile
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

class ExecutorState:
    def __init__(self, executor, token_budget: int) -> None:
        self.executor = executor
        self.tp_degree = len(executor.bundle_ids)
        self.req_ids: OrderedDict[str, None] = OrderedDict()
        self.req_ids_to_add: OrderedDict[str, None] = OrderedDict()
        self.req_ids_to_remove: OrderedDict[str, None] = OrderedDict()
        self.req_to_tokens_needed: dict[str, int] = {}
        self._token_budget = token_budget
        # TODO
        self.is_urgent = True
        # self.is_urgent = False
    
    def __repr__(self) -> str:
        return (f"ExecutorState(tp_degree={self.tp_degree}, "
                f"req_ids={self.req_ids}, "
                f"req_ids_to_add={self.req_ids_to_add}, "
                f"req_ids_to_remove={self.req_ids_to_remove}, "
                f"req_to_tokens_needed={self.req_to_tokens_needed}, "
                f"token_budget={self._token_budget})")
    
    def add_request(self, req_id: str, tokens_needed: int) -> None:
        logger.debug(f"Executor {self.executor.id} adding request {req_id}.")
        self.req_ids[req_id] = None
        if req_id in self.req_ids_to_add:
            self.req_ids_to_add.pop(req_id)
        self.req_to_tokens_needed[req_id] = tokens_needed
        logger.debug(f"after adding, executor state: {self}")

    def assert_request_added(self, req_id: str) -> None:
        assert req_id in self.req_ids
        assert req_id not in self.req_ids_to_add
        assert req_id in self.req_to_tokens_needed
    
    def remove_request(self, req_id: str) -> None:
        logger.debug(f"Executor {self.executor.id} removing request {req_id}.")
        self.req_ids.pop(req_id)
        if req_id in self.req_ids_to_remove:
            self.req_ids_to_remove.pop(req_id)
        self.req_to_tokens_needed.pop(req_id, None)
        logger.debug(f"after removing, executor state: {self}")
        
    def add_request_to_add(self, req_id: str, tokens_needed: int) -> None:
        logger.debug(f"Executor {self.executor.id} adding request to add {req_id}.")
        self.req_ids_to_add[req_id] = None
        self.req_to_tokens_needed[req_id] = tokens_needed
        logger.debug(f"after adding request to add, executor state: {self}")
    
    def add_request_to_remove(self, req_id: str) -> None:
        logger.debug(f"Executor {self.executor.id} adding request to remove {req_id}.")
        assert req_id in self.req_ids
        self.req_ids_to_remove[req_id] = None
        logger.debug(f"after adding request to remove, executor state: {self}")
        
    def get_token_budget(self) -> int:
        return self._token_budget - self.get_num_batched_tokens()
    
    def get_num_batched_tokens(self) -> int:
        return sum(
            self.req_to_tokens_needed[req_id] for req_id in self.req_ids)
        
    def get_projected_token_budget(self) -> int:
        return self.get_token_budget() - (
            sum(self.req_to_tokens_needed[req_id] for req_id in self.req_ids_to_add) - \
            sum(self.req_to_tokens_needed[req_id] for req_id in self.req_ids_to_remove)
        )

class RequestState:
    def __init__(self, request: Request) -> None:
        self.request = request
        self.executor_id = None
        self.executors_to_free: set[int] = set()
        self.pending_executor_id = None # executor to be set in the next scheduling step
        self.pred_num_steps_left = {} # confidence_threshold -> predicted steps left
        self.last_stats: StepStats = StepStats(
            id = request.request_id,
            num_denoise_ran=0,
            num_unmasked_tokens=0,
            num_cur_unmasked_tokens=0,
            output_length=request.output_length,
            block=0,
            block_num_denoise_ran=0,
            block_num_unmasked_tokens=0,
            block_size=request.denoise_block_size,
        )
        self.is_urgent = False
        self.confidence_threshold = None
    
    def __str__(self) -> str:
        return (f"RequestState(request_id={self.request.request_id}, "
                f"executor_id={self.executor_id}, "
                f"executors_to_free={self.executors_to_free}, "
                f"pending_executor_id={self.pending_executor_id}, "
                f"last_stats={self.last_stats})")
    
    def set_executor(self, executor_id: int) -> None:
        # logger.debug(f"Request {self.request.request_id} setting executor to {executor_id}")
        self.executor_id = executor_id
        self.executors_to_free.add(executor_id)
        # logger.debug(f"after setting, request state: {self}")
    
    def remove_executor(self) -> None:
        # logger.debug(f"Request {self.request.request_id} clearing executor {self.executor_id}")
        self.executors_to_free.discard(self.executor_id)
        self.executor_id = None
        # logger.debug(f"after clearing, request state: {self}")
    
    def assert_executor_set(self, executor_id: int) -> None:
        assert self.executor_id == executor_id
        assert executor_id in self.executors_to_free
        # assert self.pending_executor_id is None
    
    def set_pending_executor(self, executor_id: int) -> None:
        # logger.debug(f"Request {self.request.request_id} setting pending executor to {executor_id}")
        self.pending_executor_id = executor_id
        # logger.debug(f"after setting pending, request state: {self}")
    
    def remove_pending_executor(self) -> None:
        # logger.debug(f"Request {self.request.request_id} clearing pending executor {self.pending_executor_id}")
        self.pending_executor_id = None
        # logger.debug(f"after clearing pending, request state: {self}")
    
    def remove_executor_to_free(self, executor_id: int) -> None:
        # logger.debug(f"Request {self.request.request_id} removing executor to free {executor_id}")
        self.executors_to_free.discard(executor_id)
        # logger.debug(f"after removing, request state: {self}")
    

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
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

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
        # [tau_chang] oone kv_cache_manager per executor for now
        self.kv_cache_managers: dict[int, KVCacheManager] = {}
        # for executor_id, kv_cache_config in self.kv_cache_configs.items():
        #     self.kv_cache_managers[executor_id] = KVCacheManager(
        #         kv_cache_config=kv_cache_config,
        #         max_model_len=self.max_model_len,
        #         enable_caching=self.cache_config.enable_prefix_caching,
        #         caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
        #         use_eagle=self.use_eagle,
        #         log_stats=self.log_stats,
        #         enable_kv_cache_events=self.enable_kv_cache_events,
        #     )
        # self.kv_cache_manager = KVCacheManager(
        #     kv_cache_config=kv_cache_config,
        #     max_model_len=self.max_model_len,
        #     enable_caching=self.cache_config.enable_prefix_caching,
        #     caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
        #     use_eagle=self.use_eagle,
        #     log_stats=self.log_stats,
        #     enable_kv_cache_events=self.enable_kv_cache_events,
        # )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

        self.default_confidence_threshold = \
            self.scheduler_config.default_confidence_threshold
        self.candidate_confidence_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        self.cache_prefix = vllm_config.model_config.cache_prefix
        self.cache_suffix = vllm_config.model_config.cache_suffix

        self.step_estimator = StepEstimator(vllm_config)
        self.latency_profiles: dict[int, LatencyProfile] = {}
        # read in latency profiles
        for tp_degree in [1, 2, 4]:
            path = get_latency_profile_path(vllm_config, tp_degree)
            self.latency_profiles[tp_degree] = LatencyProfile(path)
        
        self.executors_last_updated_from_output = set()
    
    def update_step_estimates(self, req_id: str) -> None:
        req = self.request_states[req_id]
        stats_with_confidence = []
        for conf in self.candidate_confidence_thresholds:
            stats = StepStats(
                id = req.request.request_id,
                num_denoise_ran=req.last_stats.num_denoise_ran,
                num_unmasked_tokens=req.last_stats.num_unmasked_tokens,
                num_cur_unmasked_tokens=req.last_stats.num_cur_unmasked_tokens,
                output_length=req.last_stats.output_length,
                block=req.last_stats.block,
                block_num_denoise_ran=req.last_stats.block_num_denoise_ran,
                block_num_unmasked_tokens=req.last_stats.block_num_unmasked_tokens,
                block_size=req.last_stats.block_size,
                confidence_threshold=conf,
            )
            stats_with_confidence.append(stats)
            
        req.pred_num_steps_left = {
            stats.confidence_threshold: step \
                for stats, step in zip(stats_with_confidence, 
                                       self.step_estimator.\
                                           predict(stats_with_confidence))}
            
        logger.debug(f"Updated step estimates for request {req_id}: {req.pred_num_steps_left} steps left")
        
    def get_estimated_time_left(self, req_id: str, tp_degree: int, batch_size: int, confidence: float) -> float:
        pred_num_steps_left = self.request_states[req_id].pred_num_steps_left[confidence]
        ceiling_batch_size, per_step_latency = self.latency_profiles[tp_degree].lookup(batch_size)
        logger.debug(f"Estimated time left for request {req_id} with tp_degree {tp_degree} and batch_size {batch_size}: "
                     f"confidence={confidence}, "
                     f"pred_num_steps_left={pred_num_steps_left}, "
                     f"per_step_latency={per_step_latency}, "
                     f"ceiling_batch_size={ceiling_batch_size}")
        return pred_num_steps_left * per_step_latency
    
    def get_urgent_num_batched_tokens(self, executor_id: int) -> int:
        return sum(
            self.executor_states[executor_id].req_to_tokens_needed[req_id]
            for req_id in self.executor_states[executor_id].req_ids
            if self.request_states[req_id].is_urgent)
    
    def get_non_urgent_num_batched_tokens(self, executor_id: int) -> int:
        return sum(
            self.executor_states[executor_id].req_to_tokens_needed[req_id]
            for req_id in self.executor_states[executor_id].req_ids
            if not self.request_states[req_id].is_urgent)
        

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
        
        def add_new_request(request: Request, executor_id: int, is_urgent: bool,
                            confidence_threshold: float) -> None:
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

            self.executor_states[executor_id].add_request(
                request.request_id, request.num_tokens)
            self.executor_states[executor_id].is_urgent |= is_urgent

            self.request_states[request.request_id].set_executor(executor_id)
            self.request_states[request.request_id].set_pending_executor(executor_id)
            self.request_states[request.request_id].is_urgent = is_urgent
            self.request_states[request.request_id].confidence_threshold = confidence_threshold

            # if request.request_id in self.free_req_ids[executor_id]:
            #     logger.debug(f"Request {request.request_id} is newly added to executor {executor_id}, removing from free_req_ids.")
            #     self.free_req_ids[executor_id].remove(request.request_id)

            request.status = RequestStatus.RUNNING
            
        def add_running_request(request: Request, executor_id: int, is_urgent: bool,
                                confidence_threshold: float) -> None:
            scheduled_running_reqs[executor_id].append(request)
            req_to_new_block_ids[executor_id][request.request_id] = ()
            num_scheduled_tokens[executor_id][request.request_id] = \
                determine_running_exec_tokens(request)

            self.executor_states[executor_id].assert_request_added(request.request_id)
            self.executor_states[executor_id].is_urgent |= is_urgent

            self.request_states[request.request_id].assert_executor_set(executor_id)
            self.request_states[request.request_id].set_pending_executor(executor_id)
            self.request_states[request.request_id].is_urgent = is_urgent
            self.request_states[request.request_id].confidence_threshold = confidence_threshold

            request.status = RequestStatus.RUNNING
        
        def remove_request(request: Request, executor_id: int) -> None:
            # self.executor_states[executor_id].remove_request(request.request_id)
            # self.request_states[request.request_id].remove_executor()

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
            
        
        def future_add_request(request: Request, executor_id: int, is_urgent: bool, 
                               confidence_threshold: float) -> None:
            # either (inclusive) 1. request is busy 2. executor is busy
            assert self.request_states[request.request_id].pending_executor_id is None

            self.executor_states[executor_id].add_request_to_add(request.request_id, request.num_tokens)
            self.request_states[request.request_id].set_pending_executor(executor_id)
            self.request_states[request.request_id].is_urgent = is_urgent
            self.request_states[request.request_id].confidence_threshold = confidence_threshold

            request.status = RequestStatus.WAITING
        
        def future_remove_request(request: Request, executor_id: int) -> None:
            # either (inclusive) 1. request is busy 2. executor is busy
            assert self.request_states[request.request_id].executor_id == executor_id

            self.executor_states[executor_id].add_request_to_remove(request.request_id)
            self.request_states[request.request_id].set_pending_executor(None)

            request.status = RequestStatus.WAITING
            
        
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

        # update executor states
        self.update_executors()
        logger.debug(f"start of schedule, executors manager: {self.executors_manager}")
        idle_executors = []

        for executor_id in self.executors_manager.executors:
            if self.executors_manager.executors[executor_id].is_idle():
                logger.debug(f"executor {executor_id} is idle. Status transition: IDLE -> CONSIDERED_FOR_SCHEDULING")
                self.executors_manager.executors[executor_id].set_considered_for_scheduling()
                idle_executors.append(executor_id)
            else:
                logger.debug(f"executor {executor_id} is not idle. Skip.")

        # idle_executors are guranteed not killed from now on
        logger.debug(f"idle executors: {idle_executors}")
        # assert len(idle_executors) > 0

        scheduled_new_reqs: dict[int, list[Request]] = defaultdict(list)
        scheduled_resumed_reqs: dict[int, list[Request]] = defaultdict(list)
        scheduled_running_reqs: dict[int, list[Request]] = defaultdict(list)
        # this guy is not used for now
        preempted_reqs: dict[int, list[Request]] = {}

        req_to_new_block_ids: dict[int, dict[str, tuple[list[int], ...]]] = \
            defaultdict(dict) 
        num_scheduled_tokens: dict[int, dict[str, int]] = defaultdict(dict)
        
        # for executor_id in self.executors_last_updated_from_output:
        for executor_id in idle_executors:
            logger.debug(f"Processing executor {executor_id}")
            if executor_id not in idle_executors:
                logger.debug(f"executor {executor_id} is last updated, but not idle. Likely being killed. Skip.")
                continue

            tp_degree = self.executor_states[executor_id].tp_degree

            if self.executor_states[executor_id].is_urgent:
                logger.debug(f"executor {executor_id} is urgent. Start scheduling.")
                # deal with future changes first
                # assert that there is no overlap between req_ids_to_add and req_ids_to_remove
                assert len(set(self.executor_states[executor_id].req_ids_to_add) & set(self.executor_states[executor_id].req_ids_to_remove)) == 0
                
                request_newly_added = set()
                # note that req_ids_to_add changes sizes during execution
                for req_id in list(self.executor_states[executor_id].req_ids_to_add):
                    request = self.requests[req_id]
                    logger.debug(f"Adding pending request {req_id} to its executor {executor_id}")
                    if self.request_states[req_id].executor_id is None:
                        # only add if the request is done being removed.
                        add_new_request(request, executor_id, is_urgent=True, confidence_threshold=self.request_states[req_id].confidence_threshold)
                        self.running.append(request)
                        request_newly_added.add(req_id)
                    else:
                        logger.debug(f"Request {req_id} is still on executor {self.request_states[req_id].executor_id}. Cannot add to executor {executor_id} yet.")
                
                # note that req_ids_to_remove changes sizes during execution
                for req_id in list(self.executor_states[executor_id].req_ids_to_remove):
                    request = self.requests[req_id]
                    logger.debug(f"Removing pending request {req_id} from its executor {executor_id}")
                    remove_request(request, executor_id)
                    self.running = [r for r in self.running if r.request_id != req_id]
                    if self.request_states[req_id].pending_executor_id is None:
                        self.waiting.add_request(request)
                
                # A (keep request on executor if possible)
                # loop from newest to oldest requests
                logger.debug(f"Checking running requests on executor {executor_id} for SLO compliance.")
                for req_id in reversed(list(self.executor_states[executor_id].req_ids)):
                    # check slo
                    request = self.requests[req_id]
                    batch_size = self.executor_states[executor_id].get_num_batched_tokens()
                    logger.debug(f"Checking request {req_id} on executor {executor_id}. Batch size: {batch_size}")
                    is_scheduled = False
                    for conf in self.candidate_confidence_thresholds:
                        if conf < self.request_states[req_id].confidence_threshold:
                            logger.debug(f"conf {conf} < current confidence threshold {self.request_states[req_id].confidence_threshold}. Stop lowering confidence.")
                            break
                        estimated_time_left = self.get_estimated_time_left(
                            req_id, tp_degree, batch_size, self.request_states[req_id].confidence_threshold)
                        logger.debug(f"Request {req_id} with confidence {conf} est: {estimated_time_left}. SLO remaining time: {self.requests[req_id].slo_time_remaining}")
                        if estimated_time_left <= self.requests[req_id].slo_time_remaining:
                            # add running request
                            logger.debug(f"Keeping request {req_id} on executor {executor_id}")
                            if req_id not in request_newly_added:
                                add_running_request(request, executor_id, is_urgent=True, confidence_threshold=self.request_states[req_id].confidence_threshold)
                                self.running.append(request)
                                is_scheduled = True
                            break
                    if not is_scheduled:
                        logger.debug(f"Removing request {req_id} from executor {executor_id} due to SLO miss")
                        remove_request(request, executor_id)
                        self.waiting.add_request(request)
                        # self.running.remove(request)
                        self.running = [r for r in self.running if r.request_id != req_id]
            else:
                # TODO
                3/0
                

        logger.debug(f"Finished A for all idle executors. Start placing urgent requests.")
        logger.debug(f"waiting requests: {[req.request_id for req in self.waiting]}")
        # finished A for all executors
        # B
        # Place urgent
        skipped_waiting_requests = create_request_queue(self.policy)
        while self.waiting:
            request = self.waiting.pop_request()
            req_id = request.request_id
            logger.debug(f"Processing waiting request {req_id}.")
            # loop through executors, in (ascending TP_degree, descending urgent_num_batched_tokens, ascending non_urgent_num_batched_tokens) order
            is_scheduled = False
            for conf in self.candidate_confidence_thresholds:
                if is_scheduled:
                    break
                logger.debug(f"Trying to schedule urgent request {req_id} with confidence threshold {conf}.")
                for ex_id in sorted(
                    self.executors_manager.executors.keys(),
                    key=lambda eid: (
                        self.executor_states[eid].tp_degree,
                        -self.get_urgent_num_batched_tokens(eid),
                        self.get_non_urgent_num_batched_tokens(eid)
                        )
                    ):
                    if is_scheduled:
                        break

                    tp_degree = self.executor_states[ex_id].tp_degree
                    batch_size = self.get_urgent_num_batched_tokens(ex_id) + request.num_tokens
                    if batch_size > self.max_num_scheduled_tokens:
                        logger.debug(f"Can't schedule due to max_num_scheduled_tokens {self.max_num_scheduled_tokens} < {batch_size}. Skipping executor.")
                        continue
                    estimated_time_left = self.get_estimated_time_left(
                        req_id, tp_degree, batch_size, conf)
                    if estimated_time_left > request.slo_time_remaining:
                        logger.debug(f"Cannot schedule  due to SLO miss. Estimated time left: {estimated_time_left}, SLO remaining time: {request.slo_time_remaining}. Skipping executor.")
                        continue

                    logger.debug(f"Scheduling urgent request {req_id} to executor {ex_id}. Estimated time left: {estimated_time_left}, SLO remaining time: {request.slo_time_remaining}")
                    if ex_id in idle_executors:
                        # kick off non-urgent requests
                        for r_id in list(self.executor_states[ex_id].req_ids):
                            # these requests must be present
                            if not self.request_states[r_id].is_urgent:
                                r = self.requests[r_id]
                                logger.debug(f"Kicking off non-urgent request {r_id} from executor {ex_id} to make space for urgent request {req_id}")
                                remove_request(r, ex_id)
                                self.waiting.add_request(r)
                                self.running = [req for req in self.running if req.request_id != r_id]

                        add_new_request(request, ex_id, is_urgent=True, confidence_threshold=conf)
                        self.running.append(request)
                        is_scheduled = True
                    else:
                        logger.debug(f"Scheduling urgent request {req_id} to busy executor {ex_id} in the future. Estimated time left: {estimated_time_left}, SLO remaining time: {request.slo_time_remaining}")
                        # kick off non-urgent requests
                        for r_id in list(self.executor_states[ex_id].req_ids):
                            # these requests must be not present
                            if not self.request_states[r_id].is_urgent:
                                r = self.requests[r_id]
                                logger.debug(f"Kicking off non-urgent future request {r_id} from executor {ex_id} to make space for urgent request {req_id}")
                                future_remove_request(r, ex_id)
                        future_add_request(request, ex_id, is_urgent=True, confidence_threshold=conf)
                        self.running.append(request)
                        is_scheduled = True
                        
            if not is_scheduled:
                logger.debug(f"Cannot schedule urgent request {req_id} to any idle executor due to SLO miss. Skipping for now.")
                # lower confidence
                # if not possible (likely due to max_num_batch_tokens), skip for now
                skipped_waiting_requests.add_request(request)
                
        self.waiting = skipped_waiting_requests
        logger.debug(f"Finished placing urgent requests. Waiting queue: {[req.request_id for req in self.waiting]}")
        # increasing slo
        for req in self.waiting:
            req.latency_slo += 1
            logger.debug(f"Increasing SLO remaining time for waiting request {req.request_id} to {req.slo_time_remaining}")

        # # opportunistic 
        # # consider only the idle executors
        # # only consider requests on idle executors and waiting requests
        
        # cand_requests = []
        # for executor_id in idle_executors:
        #     for req_id in self.executor_states[executor_id].req_ids:
        #         cand_requests.append(self.requests[req_id])
        
        # cand_requests.extend(self.waiting.requests)
            
        # for executor_id in idle_executors:
        #     if self.executor_states[executor_id].is_urgent:
        #         logger.debug(f"Skipping opportunistic scheduling on executor {executor_id} because it has urgent requests.")
        #         continue
        #     # pick one lucky dude
        #     for req_id in cand_requests:
                
        # Construct the scheduler output.
        scheduler_outputs: dict[int, SchedulerOutput] = {}

        for executor_id in idle_executors:
            if executor_id not in num_scheduled_tokens and \
                not self.free_req_ids[executor_id]:
                logger.debug(f"Executor {executor_id} has no requests scheduled to run, and no finished requests. skip")
                # not scheduled to run at all, and no finished reqs. skip
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
        self.executors_last_updated_from_output.clear()

        return scheduler_outputs

        # First, schedule the RUNNING requests.
        logger.debug(f"start scheduling running requests: {self.running}")
        req_index = 0
        while req_index < len(self.running):
            request = self.running[req_index]
            cur_executor_id = self.request_states[request.request_id].executor_id
            logger.debug(f"considering running request {request.request_id}")

            if request.is_in_execution:
                logger.debug(f"Request {request.request_id} is in execution. Skip.")
                req_index += 1
                continue

            if cur_executor_id is None:
                logger.debug(f"Request {request.request_id}'s executor is killed or being killed. Move to waiting.")
                self.waiting.prepend_request(request)
                self.running.pop(req_index)
                continue

            if cur_executor_id not in idle_executors:
                assert cur_executor_id in self.executors_manager.executors
                logger.debug(f"Request {request.request_id}'s executor {cur_executor_id} is busy but alive. Skip.")
                req_index += 1
                continue
        
            logger.debug(f"its executor {cur_executor_id} is idle")
                
            # # request is not in execution. Assert its executor is also not
            # executor_id = self.request_to_executor[request.request_id]
            # if executor_id not in idle_executors:
            #     # this executor is being killed. Move to waiting
            #     self.waiting.prepend_request(request)
            #     self.running.pop(req_index)
            #     continue
            # assert executor_id in idle_executors, (
            #     f"Request {request.request_id} is not in execution, but its "
            #     f"executor {executor_id} is in execution")

            num_tokens_needed = request.num_tokens
            
            # logger.debug(f"in scheduler, request: {request}")
            # logger.debug(f"num_unmasked_tokens: {request.num_unmasked_tokens}")

            # if num_tokens_needed > self.executor_states[cur_executor_id].get_projected_token_budget():
            #     logger.debug(f"Not enough token budget to schedule RUNNING request {request.request_id}. Needed {num_tokens_needed}, available {self.executor_states[cur_executor_id].get_projected_token_budget()}. Skip.")
            #     logger.debug(f"executor state: {self.executor_states[cur_executor_id]}")
            #     # The request cannot be scheduled.
            #     # TODO: record this request and consider it again later.
            #     # For now just skip it.
            #     req_index += 1
            #     continue
            
            # if len(self.executor_states[cur_executor_id].req_ids) >= self.max_num_running_reqs:
            #     logger.debug(f"Executor {cur_executor_id} has reached max num running requests. Skip request {request.request_id}.")
            #     req_index += 1
            #     continue
        
            # while True:
            #     new_blocks = self.kv_cache_manager.allocate_slots(
            #         request,
            #         0,
            #         # num_new_tokens,
            #         num_lookahead_tokens=self.num_lookahead_tokens)
            #     logger.debug(f"RUNNING: new_blocks: {new_blocks} with num_tokens_needed: {num_tokens_needed}")
            #     if new_blocks is None:
            #         # The request cannot be scheduled.
            #         # Preempt the lowest-priority request.
            #         if self.policy == SchedulingPolicy.PRIORITY:
            #             preempted_req = max(
            #                 self.running,
            #                 key=lambda r: (r.priority, r.arrival_time),
            #             )
            #             self.running.remove(preempted_req)
            #         else:
            #             preempted_req = self.running.pop()

            #         self.kv_cache_manager.free(preempted_req)
            #         preempted_req.status = RequestStatus.PREEMPTED
            #         preempted_req.num_computed_tokens = 0
            #         if self.log_stats:
            #             preempted_req.record_event(
            #                 EngineCoreEventType.PREEMPTED, scheduled_timestamp)

            #         self.waiting.prepend_request(preempted_req)
            #         preempted_reqs.append(preempted_req)
            #         if preempted_req == request:
            #             # No more request to preempt.
            #             can_schedule = False
            #             break
            #     else:
            #         # The request can be scheduled.
            #         can_schedule = True
            #         break
            # if not can_schedule:
            #     break
            # assert new_blocks is not None

            # Schedule the request.
            # logger.info(f"Scheduling RUNNING request {request.request_id} ")
            # scheduled_running_reqs[executor_id].append(request)
            # # [tau_chang] no new blocks for now bc dllm
            # req_to_new_block_ids[executor_id][request.request_id] = ()

            # num_scheduled_tokens[executor_id][request.request_id] = num_tokens_needed
            # executor_load[executor_id] += num_tokens_needed
            # token_budget[executor_id] -= num_tokens_needed
            # self.request_to_executor[request.request_id] = executor_id
            # req_index += 1
            
            logger.debug(f"Confirmed: Running request {request.request_id} to executor {cur_executor_id}.")
            add_running_request(request, cur_executor_id)
            req_index += 1

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting:
                # request = self.waiting.peek_request()
                request = self.waiting.pop_request()
                logger.debug(f"considering WAITING request {request.request_id}")

                # first check if it has pending executor
                pending_executor_id = self.request_states[request.request_id].pending_executor_id
                if pending_executor_id is not None:
                    logger.debug(f"Request {request.request_id} has pending executor {pending_executor_id}")
                    if pending_executor_id not in idle_executors:
                        assert pending_executor_id in self.executors_manager.executors
                        logger.debug(f"Request {request.request_id}'s pending executor {pending_executor_id} is busy but alive. Skip.")
                        skipped_waiting_requests.prepend_request(request)
                    else:
                        logger.debug(f"Scheduling WAITING request {request.request_id} to its pending executor {pending_executor_id}")
                        add_new_request(request, pending_executor_id)
                        self.running.append(request)
                else:
                    logger.debug(f"Request {request.request_id} has no pending executor. Try to find an executor.")

                    # TODO: optimize this
                    # loop through executors in increasing order of load
                    # idle_executors.sort(key=lambda eid: self.executor_states[eid].get_projected_token_budget(), reverse=True)
                    all_executors = sorted(
                        self.executor_states.keys(),
                        key=lambda eid: self.executor_states[eid].get_projected_token_budget(), 
                        reverse=True)
                    
                    for executor_id in all_executors:
                        # if executor_id in scheduled_running_reqs and \
                        #     len(scheduled_running_reqs[executor_id]) == \
                        #         self.max_num_running_reqs:
                        #     continue

                        if request.num_tokens > self.executor_states[executor_id].get_projected_token_budget():
                            logger.debug(f"Not enough token budget to schedule WAITING request {request.request_id} to executor {executor_id}. Needed {request.num_tokens}, available {self.executor_states[executor_id].get_projected_token_budget()}. Skip.")
                            skipped_waiting_requests.prepend_request(request)
                            break

                        if len(self.executor_states[executor_id].req_ids) >= self.max_num_running_reqs:
                            logger.debug(f"Executor {executor_id} has reached max num running requests. Try another executor.")
                            continue

                        if executor_id in idle_executors:
                            logger.debug(f"Confirmed: Scheduling WAITING request {request.request_id} to executor {executor_id}.")

                            add_new_request(request, executor_id)
                            self.running.append(request)
                            break
                        else:
                            logger.debug(f"Confirmed: Scheduling WAITING request {request.request_id} to executor {executor_id} as pending.")
                            mark_request_as_pending(request, executor_id)
                            skipped_waiting_requests.prepend_request(request)
                            break
                    else:
                        logger.debug(f"Request {request.request_id} cannot be scheduled to any executor now. Skip.")
                        skipped_waiting_requests.prepend_request(request)
                        

                    #     num_external_computed_tokens = 0

                    #     num_tokens_needed = request.num_tokens
                    #     if num_tokens_needed > token_budget[executor_id]:
                    #         # The request cannot be scheduled.
                    #         continue
                    #         # self.waiting.pop_request()
                    #         # skipped_waiting_requests.prepend_request(request)
                    #         # continue

                    #     new_blocks = self.kv_cache_managers[executor_id].allocate_slots(
                    #         request,
                    #         num_tokens_needed,
                    #     )
                    #     if new_blocks is None:
                    #         # The request cannot be scheduled.
                    #         continue
                    
                    #     logger.debug(f"WAITING: new_blocks: {new_blocks} with num_tokens_needed: {num_tokens_needed}")
                            
                    #     # Request was already popped from self.waiting
                    #     # unless it was re-added above due to new_blocks being None.
                    #     request = self.waiting.pop_request()
                    #     req_index += 1 # not used ?
                    #     self.running.append(request)
                    #     if self.log_stats:
                    #         request.record_event(EngineCoreEventType.SCHEDULED,
                    #                             scheduled_timestamp)
                    #     if request.status == RequestStatus.WAITING:
                    #         scheduled_new_reqs[executor_id].append(request)
                    #     elif request.status == RequestStatus.PREEMPTED:
                    #         scheduled_resumed_reqs[executor_id].append(request)
                    #     else:
                    #         raise RuntimeError(
                    #             f"Invalid request status: {request.status}")

                    #     req_to_new_block_ids[executor_id][request.request_id] = \
                    #             self.kv_cache_managers[executor_id].get_block_ids(
                    #                 request.request_id)
                    #     num_scheduled_tokens[executor_id][request.request_id] = num_tokens_needed
                    #     executor_load[executor_id] += num_tokens_needed
                    #     token_budget[executor_id] -= num_tokens_needed
                    #     self.request_to_executor[request.request_id] = executor_id
                    #     request.status = RequestStatus.RUNNING

                    #     # no need to look for other executors
                    #     break
                    # else:
                    #     # The request cannot be scheduled.
                    #     request = self.waiting.pop_request()
                    #     skipped_waiting_requests.prepend_request(request)
                    #     continue

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check constraints per executor
        for executor_id in idle_executors:
            # check token budget
            assert self.executor_states[executor_id].get_token_budget() >= 0
            if executor_id in scheduled_running_reqs:
                assert len(scheduled_running_reqs[executor_id]) <= \
                    self.max_num_running_reqs
            if executor_id in num_scheduled_tokens:
                total_scheduled = sum(num_scheduled_tokens[executor_id].values())
                assert total_scheduled <= self.max_num_scheduled_tokens

        # # Check if the scheduling constraints are satisfied.
        # total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        # assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        # assert token_budget >= 0
        # assert len(self.running) <= self.max_num_running_reqs
        # # Since some requests in the RUNNING queue may not be scheduled in
        # # this step, the total number of scheduled requests can be smaller than
        # # len(self.running).
        # assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
        #         len(scheduled_running_reqs) <= len(self.running))

        # Construct the scheduler output.
        scheduler_outputs: dict[int, SchedulerOutput] = {}

        for executor_id in idle_executors:
            if executor_id not in num_scheduled_tokens and \
                not self.free_req_ids[executor_id]:
                logger.debug(f"Executor {executor_id} has no requests scheduled to run, and no finished requests. skip")
                # not scheduled to run at all, and no finished reqs. skip
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
                    confidence_thresholds[req_id] = self.default_confidence_threshold
                    logger.debug(f"using default confidence threshold {confidence_thresholds[req_id]} for req {req_id}")
            
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
        # new_reqs_data = [
        #     NewRequestData.from_request(req, 
        #                                 req_to_new_block_ids[req.request_id])
        #     for req in scheduled_new_reqs
        # ]
        # cached_reqs_data = self._make_cached_request_data(
        #     scheduled_running_reqs,
        #     scheduled_resumed_reqs,
        #     num_scheduled_tokens,
        #     scheduled_spec_decode_tokens,
        #     req_to_new_block_ids,
        # )
        # scheduler_output = SchedulerOutput(
        #     scheduled_new_reqs=new_reqs_data,
        #     scheduled_cached_reqs=cached_reqs_data,
        #     num_scheduled_tokens=num_scheduled_tokens,
        #     total_num_scheduled_tokens=total_num_scheduled_tokens,
        #     scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        #     scheduled_encoder_inputs=scheduled_encoder_inputs,
        #     num_common_prefix_blocks=[],
        #     # finished_req_ids is an existing state in the scheduler,
        #     # instead of being newly scheduled in this step.
        #     # It contains the request IDs that are finished in between
        #     # the previous and the current steps.
        #     finished_req_ids=self.finished_req_ids,
        #     free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
        #     structured_output_request_ids=structured_output_request_ids,
        #     grammar_bitmask=None,
        # )

        # self._update_after_schedule(0, scheduler_output)
        logger.debug(f"returning scheduler output: {scheduler_outputs}")
        # return scheduler_output
        # [TODO (tau_chang)]: update this
        # return {0: scheduler_output}
        self.executors_last_updated_from_output.clear()

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

        self.executors_last_updated_from_output.add(executor_id)
        
        async with self.executors_manager.cond[executor_id]:
            logger.debug(f"In update_from_output, executor {executor_id} status transition: OUTPUT_READY -> IDLE")
            self.executors_manager.executors[executor_id].set_idle()
            self.executors_manager.cond[executor_id].notify()

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
                stats, new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, 
                    scheduler_output.confidence_thresholds[req_id], 
                    model_runner_output.avg_output_confidences[req_index])
                stats.confidence_threshold = \
                    scheduler_output.confidence_thresholds[req_id]
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
        if stopped_running_reqs:
            self.running = [
                req for req in self.running if req not in stopped_running_reqs
            ]
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

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

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[tuple[int, int]],
        confidence_threshold: float,
        avg_output_confidence: float,
    ) -> tuple[list[tuple[int, int]], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        stats = request.update_from_output(
            new_token_ids,
            confidence_threshold,
            avg_output_confidence,
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
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        logger.debug(f"Scheduler adding request {request.request_id}")
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        self.request_states[request.request_id] = RequestState(request)
        self.update_step_estimates(request.request_id)
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

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
        for request in running_requests_to_remove:
            self.running.remove(request)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request, True)
        
    def _free_request_on_executor(self, request: Request, executor_id: int, 
                      finished: bool, prune: bool = True) -> Optional[dict[str, Any]]:
        logger.debug(f"in _free_request_on_executor, request: {request}, executor_id: {executor_id}, finished: {finished}")
        request_id = request.request_id
        assert executor_id in self.request_states[request_id].executors_to_free

        # # [tau_chang] Not used start
        # delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        # self.encoder_cache_manager.free(request)
        # # [tau_chang] Not used end

        self.free_req_ids[executor_id].add(request_id)
        if finished:
            self.finished_req_ids[executor_id].add(request_id)

            if self.finished_req_ids_dict is not None:
                self.finished_req_ids_dict[executor_id][request.client_index].\
                    add(request_id)

        # if not delay_free_blocks:
        #     self._free_blocks(request, executor_id)
        self._free_blocks(request, executor_id)

        # update states
        if executor_id == self.request_states[request_id].executor_id:
            self.request_states[request_id].remove_executor()
        else:
            self.request_states[request_id].remove_executor_to_free(executor_id)
        self.request_states[request_id].remove_pending_executor()

        self.executor_states[executor_id].remove_request(request_id)
        if prune and not self.request_states[request_id].executors_to_free:
            del self.request_states[request_id]
            del self.requests[request_id]
        
        return None
        

    def _free_request(self, request: Request, 
                      finished: bool) -> Optional[dict[str, Any]]:
        logger.debug(f"in _free_request, request: {request}, finished: {finished}")
        for executor_id in list(self.request_states[request.request_id].executors_to_free):
            self._free_request_on_executor(request, executor_id, finished)
        
        self.step_estimator.save_data()
        return None


    def _free_blocks(self, request: Request, executor_id: int):
        # assert request.is_finished()
        # executor_id = self.request_to_executor.get(request.request_id)
        self.kv_cache_managers[executor_id].free(request)
        self.kv_cache_managers[executor_id].free_block_hashes(request)
        logger.debug(f"Freed blocks for request {request.request_id} on executor {executor_id}")
        # del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        # logger.debug(f"in has_finished_requests, finished_req_ids: {self.finished_req_ids}")
        # return any(len(finished) > 0 for finished in self.finished_req_ids.values())
        logger.debug(f"in has_finished_requests, free_req_ids: {self.free_req_ids}")
        return any(len(free) > 0 for free in self.free_req_ids.values())
        # return len(self.finished_req_ids) > 0

    # def add_request(self, request: Request) -> None:
    #     self.waiting.add_request(request)
    #     self.requests[request.request_id] = request
    #     self.request_states[request.request_id] = RequestState(request)
    #     if self.log_stats:
    #         request.record_event(EngineCoreEventType.QUEUED)

    # def finish_requests(
    #     self,
    #     request_ids: Union[str, Iterable[str]],
    #     finished_status: RequestStatus,
    # ) -> None:
    #     """Handles the finish signal from outside the scheduler.

    #     For example, the API server can abort a request when the client
    #     disconnects.
    #     """
    #     assert RequestStatus.is_finished(finished_status)
    #     if isinstance(request_ids, str):
    #         request_ids = (request_ids, )
    #     else:
    #         request_ids = set(request_ids)

    #     running_requests_to_remove = []
    #     waiting_requests_to_remove = []
    #     valid_requests = []

    #     # First pass: collect requests to remove from queues
    #     for req_id in request_ids:
    #         request = self.requests.get(req_id)
    #         if request is None:
    #             # Invalid request ID.
    #             continue

    #         valid_requests.append(request)
    #         if request.status == RequestStatus.RUNNING:
    #             running_requests_to_remove.append(request)
    #         else:
    #             waiting_requests_to_remove.append(request)

    #     # Remove all requests from queues at once for better efficiency
    #     for request in running_requests_to_remove:
    #         self.running.remove(request)
    #     if waiting_requests_to_remove:
    #         self.waiting.remove_requests(waiting_requests_to_remove)

    #     # Second pass: set status and free requests
    #     for request in valid_requests:
    #         request.status = finished_status
    #         self._free_request(request)

    # def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
    #     assert request.is_finished()
    #     executor_id = self.request_to_executor[request.request_id]

    #     delay_free_blocks, kv_xfer_params = self._connector_finished(request)
    #     self.encoder_cache_manager.free(request)
    #     request_id = request.request_id
    #     self.finished_req_ids[executor_id].add(request_id)
    #     if self.finished_req_ids_dict is not None:
    #         self.finished_req_ids_dict[executor_id][request.client_index].\
    #             add(request_id)

    #     if not delay_free_blocks:
    #         self._free_blocks(request)
        
    #     self.request_to_executor.pop(request_id, None)

    #     return kv_xfer_params

    # def _free_blocks(self, request: Request):
    #     assert request.is_finished()
    #     executor_id = self.request_to_executor.get(request.request_id)
    #     self.kv_cache_managers[executor_id].free(request)
    #     self.kv_cache_managers[executor_id].free_block_hashes(request)
    #     logger.debug(f"Freed blocks for request {request.request_id} on executor {executor_id}")
    #     del self.requests[request.request_id]

    # def get_num_unfinished_requests(self) -> int:
    #     return len(self.waiting) + len(self.running)

    # def has_finished_requests(self) -> bool:
    #     logger.debug(f"in has_finished_requests, finished_req_ids: {self.finished_req_ids}")
    #     return any(len(finished) > 0 for finished in self.finished_req_ids.values())
    #     # return len(self.finished_req_ids) > 0
    
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
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_managers[executor_id].usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted
                                   for req in self.running),
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
        logger.debug(f"Added executor {executor.id} to scheduler.")
            
    def remove_executor(self, executor_id: int) -> None:
        assert executor_id in self.executor_states
        for req_id in self.executor_states[executor_id].req_ids:
            self.request_states[req_id].remove_executor()
        for req_id in self.executor_states[executor_id].pending_req_ids:
            self.request_states[req_id].remove_pending_executor()

        del self.executor_states[executor_id]
        logger.debug(f"Removed executor {executor_id} from scheduler.")

    
    def update_executors(self) -> None:
        for executor_id in self.executors_manager.executors:
            if executor_id not in self.executor_states:
                self.add_executor(executor_id)
        
        for executor_id in list(self.executor_states.keys()):
            if executor_id not in self.executors_manager.executors or \
                self.executors_manager.executors[executor_id].is_killing():
                self.remove_executor(executor_id)