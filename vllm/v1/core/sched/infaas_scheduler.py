# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import os
import time
from collections import defaultdict
from collections.abc import Iterable
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
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.step_estimator import StepEstimator, StepStats
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop, SystemLogger
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.executors_manager import ExecutorsManager
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import TimeProfiler

logger = init_logger(__name__)

class ExecutorState:
    def __init__(self, executor, token_budget: int) -> None:
        self.executor = executor
        self.executor_id = executor.id # for logging
        self.tp_degree = len(executor.bundle_ids)
        self.req_ids: set[str] = set()
        self.pending_req_ids: set[str] = set()
        self.req_to_tokens_needed: dict[str, int] = {}
        self._token_budget = token_budget
    
    def __repr__(self) -> str:
        return (f"ExecutorState(tp_degree={self.tp_degree}, "
                f"req_ids={self.req_ids}, "
                f"pending_req_ids={self.pending_req_ids}, "
                f"req_to_tokens_needed={self.req_to_tokens_needed}, "
                f"token_budget={self._token_budget})")
    
    def add_request(self, req_id: str, tokens_needed: int) -> None:
        logger.debug(f"Executor {self.executor.id} adding request {req_id}.")
        self.req_ids.add(req_id)
        if req_id in self.pending_req_ids:
            self.pending_req_ids.discard(req_id)
        self.req_to_tokens_needed[req_id] = tokens_needed
        logger.debug(f"after adding, executor state: {self}")

    def assert_request_added(self, req_id: str) -> None:
        assert req_id in self.req_ids
        assert req_id not in self.pending_req_ids
        assert req_id in self.req_to_tokens_needed
    
    def remove_request(self, req_id: str) -> None:
        logger.debug(f"Executor {self.executor.id} removing request {req_id}.")
        self.req_ids.discard(req_id)
        self.req_to_tokens_needed.pop(req_id, None)
        logger.debug(f"after removing, executor state: {self}")
        
    def add_pending_request(self, req_id: str, tokens_needed) -> None:
        logger.debug(f"Executor {self.executor.id} adding pending request {req_id}.")
        self.pending_req_ids.add(req_id)
        self.req_to_tokens_needed[req_id] = tokens_needed
        logger.debug(f"after adding pending, executor state: {self}")
        
    def get_token_budget(self) -> int:
        tokens_in_use = sum(
            self.req_to_tokens_needed[req_id] for req_id in self.req_ids)
        return self._token_budget - tokens_in_use
    
    def get_projected_token_budget(self) -> int:
        return self.get_token_budget() - sum(
            self.req_to_tokens_needed[req_id] for req_id in self.pending_req_ids)
    
    def is_drained(self) -> bool:
        return len(self.req_ids) == 0 and len(self.pending_req_ids) == 0

class RequestState:
    def __init__(self, request: Request) -> None:
        self.request = request
        self.request_id = request.request_id # for logging
        self.executor_id = None
        self.executors_to_free: set[int] = set()
        self.pending_executor_id = None
        self.confidence_threshold = None # for logging
    
    def __str__(self) -> str:
        return (f"RequestState(request_id={self.request.request_id}, "
                f"executor_id={self.executor_id}, "
                f"executors_to_free={self.executors_to_free}, "
                f"pending_executor_id={self.pending_executor_id})")
    
    def set_executor(self, executor_id: int) -> None:
        logger.debug(f"Request {self.request.request_id} setting executor to {executor_id}")
        self.executor_id = executor_id
        self.executors_to_free.add(executor_id)
        self.pending_executor_id = None
        logger.debug(f"after setting, request state: {self}")
    
    def remove_executor(self) -> None:
        logger.debug(f"Request {self.request.request_id} clearing executor {self.executor_id}")
        self.executors_to_free.discard(self.executor_id)
        self.executor_id = None
        self.pending_executor_id = None
        logger.debug(f"after clearing, request state: {self}")
    
    def assert_executor_set(self, executor_id: int) -> None:
        assert self.executor_id == executor_id
        assert executor_id in self.executors_to_free
        assert self.pending_executor_id is None
    
    def set_pending_executor(self, executor_id: int) -> None:
        logger.debug(f"Request {self.request.request_id} setting pending executor to {executor_id}")
        self.pending_executor_id = executor_id
        logger.debug(f"after setting pending, request state: {self}")
    
    def remove_pending_executor(self) -> None:
        logger.debug(f"Request {self.request.request_id} clearing pending executor {self.pending_executor_id}")
        self.pending_executor_id = None
        logger.debug(f"after clearing pending, request state: {self}")
    
    def remove_executor_to_free(self, executor_id: int) -> None:
        logger.debug(f"Request {self.request.request_id} removing executor to free {executor_id}")
        self.executors_to_free.discard(executor_id)
        logger.debug(f"after removing, request state: {self}")

class InFaaSScheduler(SchedulerInterface):

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

        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0

        # Create the KV cache manager.
        # [tau_chang] oone kv_cache_manager per executor for now
        self.kv_cache_managers: dict[int, KVCacheManager] = {}
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

        self.default_confidence_threshold = \
            self.scheduler_config.default_confidence_threshold
        
        self.cache_prefix = vllm_config.model_config.cache_prefix
        self.cache_suffix = vllm_config.model_config.cache_suffix

        self.step_estimator = StepEstimator(vllm_config)

        self.system_logger = SystemLogger("system_logs", self)
        self.system_logger.log()

        profiler_path = os.path.join(
            self.vllm_config.experiment_config.experiment_dir,
            "profiles/scheduler/schedule.jsonl")
        self.schedule_profiler = TimeProfiler("Schedule", profiler_path)
        profiler_path = os.path.join(
            self.vllm_config.experiment_config.experiment_dir,
            "profiles/scheduler/update.jsonl")
        self.update_profiler = TimeProfiler("Update", profiler_path)
        
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
            self.executor_states[executor_id].add_request(
                request.request_id, request.num_tokens
            )
            self.request_states[request.request_id].set_executor(executor_id)

            request.status = RequestStatus.RUNNING
            
        def add_running_request(request: Request, executor_id: int) -> None:
            scheduled_running_reqs[executor_id].append(request)
            req_to_new_block_ids[executor_id][request.request_id] = ()
            num_scheduled_tokens[executor_id][request.request_id] = \
                determine_running_exec_tokens(request)

            self.executor_states[executor_id].assert_request_added(request.request_id)
            self.request_states[request.request_id].assert_executor_set(executor_id)

            request.status = RequestStatus.RUNNING
        
        def mark_request_as_pending(request: Request, executor_id: int) -> None:
            self.executor_states[executor_id].add_pending_request(
                request.request_id, request.num_tokens)
            self.request_states[request.request_id].set_pending_executor(
                executor_id)
            
            request.status = RequestStatus.WAITING

        with self.schedule_profiler.section("total"):
            with self.schedule_profiler.section("update executors"):
                # update executor states
                self.update_executors()
            with self.schedule_profiler.section("init"):
                logger.debug(f"start of schedule, executors manager: {self.executors_manager}")
                # obtain lock
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

                scheduled_new_reqs: dict[int, list[Request]] = defaultdict(list)
                scheduled_resumed_reqs: dict[int, list[Request]] = defaultdict(list)
                scheduled_running_reqs: dict[int, list[Request]] = defaultdict(list)
                # this guy is not used for now
                preempted_reqs: dict[int, list[Request]] = {}

                # NOTE: structured_output_request_ids maps
                # a request's (request that uses structured output)
                # request_id to the running request index.
                # This will helps us determine to slice the grammar bitmask
                # and only applies valid mask for requests that
                # uses structured decoding.
                # structured_output_request_ids: dict[str, int] = {}
                structured_output_request_ids: dict[int, dict[str, int]] = {}

                # req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
                # num_scheduled_tokens: dict[str, int] = {}
                # token_budget = self.max_num_scheduled_tokens
                req_to_new_block_ids: dict[int, dict[str, tuple[list[int], ...]]] = \
                    defaultdict(dict) 
                num_scheduled_tokens: dict[int, dict[str, int]] = defaultdict(dict)

            with self.schedule_profiler.section("running_requests"):
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

                    if cur_executor_id not in idle_executors + idle_not_accepting_executors:
                        assert cur_executor_id in self.executors_manager.executors
                        logger.debug(f"Request {request.request_id}'s executor {cur_executor_id} is busy but alive. Skip.")
                        req_index += 1
                        continue
                
                    logger.debug(f"its executor {cur_executor_id} is idle")
                        
                    logger.debug(f"Confirmed: Running request {request.request_id} to executor {cur_executor_id}.")
                    add_running_request(request, cur_executor_id)
                    req_index += 1

                # Use a temporary RequestQueue to collect requests that need to be
                # skipped and put back at the head of the waiting queue later
                skipped_waiting_requests = create_request_queue(self.policy)

            with self.schedule_profiler.section("waiting_requests"):
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
                            if pending_executor_id not in idle_executors + idle_not_accepting_executors:
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
                                key=lambda eid: (
                                    self.executor_states[eid].get_projected_token_budget(),
                                    self.executor_states[eid].tp_degree,
                                ),
                                reverse=True)
                            
                            logger.debug(f"all executors sorted by projected token budget and tp degree: {all_executors}")
                            logger.debug(f"token budgets: {{eid: self.executor_states[eid].get_projected_token_budget() for eid in all_executors}}")
                            logger.debug(f"TP degrees: {{eid: self.executor_states[eid].tp_degree for eid in all_executors}}")
                            
                            for executor_id in all_executors:
                                if executor_id in idle_not_accepting_executors:
                                    logger.debug(f"Executor {executor_id} is idle but not accepting new requests. Skip.")
                                    continue

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
                                

                # Put back any skipped requests at the head of the waiting queue
                if skipped_waiting_requests:
                    self.waiting.prepend_requests(skipped_waiting_requests)

            with self.schedule_profiler.section("create_outputs"):
                # Check constraints per executor
                for executor_id in idle_executors + idle_not_accepting_executors:
                    # check token budget
                    assert self.executor_states[executor_id].get_token_budget() >= 0
                    if executor_id in scheduled_running_reqs:
                        assert len(scheduled_running_reqs[executor_id]) <= \
                            self.max_num_running_reqs
                    if executor_id in num_scheduled_tokens:
                        total_scheduled = sum(num_scheduled_tokens[executor_id].values())
                        assert total_scheduled <= self.max_num_scheduled_tokens

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
                        free_encoder_input_ids=[],
                        structured_output_request_ids=structured_output_request_ids.get(executor_id, {}),
                        grammar_bitmask=None,
                        confidence_thresholds=confidence_thresholds
                    )
                    scheduler_outputs[executor_id] = scheduler_output

                    self._update_after_schedule(executor_id, scheduler_output)

                    self.system_logger.log()

        logger.debug(f"returning scheduler output: {scheduler_outputs}")
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

    async def update_from_output(
        self,
        executor_id: int,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None
        
        with self.update_profiler.section("total"):
            with self.update_profiler.section("request_loop"):
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

                    stopped = False
                    new_logprobs = None
                    new_token_ids = generated_token_ids
                    kv_transfer_params = None
                    status_before_stop = request.status

                    # Check for stop and update request status.
                    if new_token_ids:
                        with self.update_profiler.event("update_request_with_output",
                                                        req_id=req_id):
                            stats, new_token_ids, stopped = self._update_request_with_output(
                                request, new_token_ids, 
                                scheduler_output.confidence_thresholds[req_id], 
                                model_runner_output.confidence_stats[req_index])
                        with self.update_profiler.event("add_data_point",
                                                        req_id=req_id):
                            self.step_estimator.add_data_point(stats)
                        
                    with self.update_profiler.event("rest",
                                                    req_id=req_id):
                        if stopped:
                            kv_transfer_params = self._free_request(request, True)
                            if status_before_stop == RequestStatus.RUNNING:
                                stopped_running_reqs.add(request)
                            else:
                                stopped_preempted_reqs.add(request)

                        # Add EngineCoreOutput for this Request.
                        outputs[request.client_index].append(
                            EngineCoreOutput(
                                request_id=req_id,
                                new_token_ids=new_token_ids,
                                finish_reason=request.get_finished_reason(),
                                new_logprobs=new_logprobs,
                                new_prompt_logprobs_tensors=None,
                                pooling_output=None,
                                stop_reason=request.stop_reason,
                                events=request.take_events(),
                                kv_transfer_params=kv_transfer_params,
                                num_cached_tokens=request.num_cached_tokens,
                                num_denoise_ran=request.num_denoise_ran,
                            ))

                # Remove the stopped requests from the running and waiting queues.
                if stopped_running_reqs:
                    self.running = [
                        req for req in self.running if req not in stopped_running_reqs
                    ]
                if stopped_preempted_reqs:
                    # This is a rare case and unlikely to impact performance.
                    self.waiting.remove_requests(stopped_preempted_reqs)

                # # KV Connector: update state for finished KV Transfers.
                # self._update_from_kv_xfer_finished(model_runner_output)

            with self.update_profiler.section("create_engine_core_outputs"):
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


            with self.update_profiler.section("set_executor_idle"):
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

        self.update_profiler.add_info("executor_id", executor_id)
        self.update_profiler.add_info("num_requests", len(num_scheduled_tokens))
        
        self.update_profiler.commit()
        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[tuple[int, int]],
        confidence_threshold: float,
        confidence_stats: dict[str, int],
        max_confidence_threshold: Optional[float] = None,
    ) -> tuple[list[tuple[int, int]], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        stats = request.update_from_output(
            new_token_ids,
            confidence_threshold,
            confidence_stats,
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

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        logger.debug(f"Scheduler adding request {request.request_id}")
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        self.request_states[request.request_id] = RequestState(request)
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)
        self.system_logger.log()
    
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

        self.executor_states[executor_id].remove_request(request_id)
        if prune and not self.request_states[request_id].executors_to_free:
            del self.request_states[request_id]
            del self.requests[request_id]
            self.system_logger.log()
        
        return None
        

    def _free_request(self, request: Request, 
                      finished: bool) -> Optional[dict[str, Any]]:
        logger.debug(f"in _free_request, request: {request}, finished: {finished}")
        for executor_id in list(self.request_states[request.request_id].executors_to_free):
            self._free_request_on_executor(request, executor_id, finished)
        
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

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()

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
        del self.free_req_ids[executor_id]
        logger.debug(f"Removed executor {executor_id} from scheduler.")

    
    def update_executors(self) -> None:
        for executor_id in self.executors_manager.executors:
            if executor_id not in self.executor_states:
                self.add_executor(executor_id)
        
        for executor_id in list(self.executor_states.keys()):
            if executor_id not in self.executors_manager.executors or \
                self.executors_manager.executors[executor_id].is_killing():
                self.remove_executor(executor_id)