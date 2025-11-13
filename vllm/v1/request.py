# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from typing import TYPE_CHECKING, Any, Optional, Union

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import is_list_of
from vllm.v1.core.sched.step_estimator import StepStats
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest

logger = init_logger(__name__)

class Request:

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        mask_token_id: int,
        multi_modal_inputs: Optional[list[MultiModalKwargs]],
        multi_modal_hashes: Optional[list[str]],
        multi_modal_placeholders: Optional[list[PlaceholderRange]],
        sampling_params: Optional[SamplingParams],
        pooling_params: Optional[PoolingParams],
        eos_token_id: Optional[int],
        client_index: int = 0,
        arrival_time: Optional[float] = None,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
        cache_salt: Optional[str] = None,
        priority: int = 0,
        denoise_block_size: int = -1,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        # [tau_chang] for now
        self.sampling_params.min_tokens = sampling_params.max_tokens
        assert self.sampling_params.min_tokens == self.sampling_params.max_tokens

        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id

        self.lora_request = lora_request
        self.structured_output_request = structured_output_request
        self.arrival_time = arrival_time if arrival_time is not None else \
            time.time()

        self.status = RequestStatus.WAITING
        if sampling_params and sampling_params.guided_decoding is not None:
            self.status = RequestStatus.WAITING_FOR_FSM
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: Union[int, str, None] = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: Optional[dict[str, Any]] = None

        if pooling_params is not None:
            self.max_tokens = 1
        elif sampling_params is not None:
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if sampling_params.guided_decoding is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = \
                    sampling_params.extra_args.get("kv_transfer_params")
        else:
            raise ValueError(
                "sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)

        self.output_length = self.max_tokens

        self._unmasked_token_ids: list[tuple[int, int]] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self._all_token_ids.extend([mask_token_id] * self.output_length)

        self._is_in_execution = False

        self.num_output_placeholders = 0  # Used in async scheduling.
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: Optional[str] = cache_salt

        self.num_last_unmasked_tokens = 0
        self.num_denoise_ran = 0

        # [exec_start_pos, num_exec_tokens) fed into the model for execution.
        # Might change in each iteration due to caching.
        self.exec_start_pos = 0
        self.num_exec_tokens = len(self._all_token_ids)

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_inputs = multi_modal_inputs or []
        self.mm_hashes: list[str] = multi_modal_hashes or []
        self.num_encoder_inputs = len(self.mm_inputs)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Sanity check
        assert len(self.mm_inputs) == len(self.mm_positions)
        if self.mm_hashes:
            assert len(self.mm_inputs) == len(self.mm_hashes)
        
        # for block diffusion
        self.denoise_block_size = denoise_block_size if denoise_block_size > 0 \
            else self.output_length
        assert self.output_length % self.denoise_block_size == 0
        self.cur_block = 0
        self.cur_block_start = len(self.prompt_token_ids)
        self.cur_block_num_unmasked_tokens = 0
        self.cur_block_denoise_ran = 0

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.all_token_ids = ConstantList(self._all_token_ids)
        self.unmasked_token_ids = ConstantList(self._unmasked_token_ids)

        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        self.last_recompute_avg_output_confidence = None
        # TODO: fix SLO
        self.latency_slo = 3.2


    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest,
                                 mask_token_id: int,
                                 denoise_block_size: int) -> "Request":
        if request.mm_inputs is not None:
            assert isinstance(request.mm_inputs, list)
            assert is_list_of(request.mm_inputs, MultiModalKwargs), (
                "mm_inputs was not updated in EngineCore.add_request")

        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            mask_token_id=mask_token_id,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=StructuredOutputRequest(
                sampling_params=request.sampling_params) \
                    if request.sampling_params else None,
            cache_salt=request.cache_salt,
            priority=request.priority,
            denoise_block_size=denoise_block_size,
        )

    def update_from_output(
        self,
        token_ids: list[tuple[int, int]],
        confidence_threshold: float,
        avg_output_confidence: float,
    ) -> StepStats:
    
        no_cache = self.num_exec_tokens == len(self._all_token_ids)
        if no_cache or self.is_start_of_new_block:
            self.last_recompute_avg_output_confidence = avg_output_confidence

        self.num_last_unmasked_tokens = len(token_ids)
        self._unmasked_token_ids.extend(token_ids)
        self.cur_block_num_unmasked_tokens += len(token_ids)
        logger.debug(f"Request {self.request_id} has unmasked {self.num_unmasked_tokens} tokens so far.")
        logger.debug(f"Request {self.request_id} has unmasked {self.cur_block_num_unmasked_tokens} tokens in the current block (block {self.cur_block}).")
        logger.debug(f"get {len(token_ids)} token_ids: {token_ids}")
        assert self.cur_block_num_unmasked_tokens <= self.denoise_block_size

        
        stats = StepStats(
            id=self.request_id,
            num_denoise_ran=self.num_denoise_ran,
            num_unmasked_tokens=len(self._unmasked_token_ids),
            num_cur_unmasked_tokens=self.num_last_unmasked_tokens,
            output_length=self.output_length,
            block=self.cur_block,
            block_num_denoise_ran=self.cur_block_denoise_ran,
            block_num_unmasked_tokens=self.cur_block_num_unmasked_tokens,
            block_size=self.denoise_block_size,
            confidence_threshold=confidence_threshold,
            last_recompute_avg_output_confidence=self.last_recompute_avg_output_confidence,
            cur_avg_output_confidence=avg_output_confidence,
        )

        self.num_denoise_ran += 1
        self.cur_block_denoise_ran += 1
        old_cur_block_start = self.cur_block_start
        
        if self.cur_block_num_unmasked_tokens == self.denoise_block_size:
            # move to the next block
            logger.debug(f"Request {self.request_id} finished denoising block starting at position {self.cur_block_start}. Moving to next block.")
            self.cur_block += 1
            self.cur_block_start += self.denoise_block_size
            self.cur_block_num_unmasked_tokens = 0
            self.cur_block_denoise_ran = 0

        for pos, token_id in token_ids:
            logger.debug(f"Request {self.request_id} unmasked token {token_id} at position {pos}.")
            logger.debug(f"cur block: {old_cur_block_start} to {old_cur_block_start + self.denoise_block_size - 1}")
            assert pos >= old_cur_block_start and pos < old_cur_block_start + self.denoise_block_size

            pos -= len(self.prompt_token_ids)
            self._all_token_ids[pos] = token_id
        
        return stats
            

    @property
    def is_output_corrupted(self) -> bool:
        return self.num_nans_in_logits > 0

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)
    
    @property
    def num_unmasked_tokens(self) -> int:
        return len(self._unmasked_token_ids)
    
    @property
    def unmask_progress(self) -> float:
        return self.num_unmasked_tokens / self.output_length
    
    @property
    def is_in_execution(self) -> bool:
        return self._is_in_execution
    
    @property
    def num_recompute_steps_left(self) -> int:
        total_blocks = self.output_length // self.denoise_block_size
        num_remaining_blocks = total_blocks - self.cur_block
        return num_remaining_blocks - (not self.is_start_of_new_block)
    
    def set_in_execution(self, in_execution: bool) -> None:
        """Set the execution state of the request."""
        self._is_in_execution = in_execution

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id].length
        return num_tokens
    
    
    @property
    def is_start_of_new_block(self) -> bool:
        # logger.debug(f"self.cur_block_num_unmasked_tokens: {self.cur_block_num_unmasked_tokens}")
        return self.cur_block_num_unmasked_tokens == 0

    @property
    def use_structured_output(self) -> bool:
        return self.sampling_params is not None and \
            self.sampling_params.guided_decoding is not None

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: Optional[float] = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> Optional[list[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    @property
    def slo_time_remaining(self) -> float:
        if self.latency_slo is None:
            return float("inf")
        elapsed = time.time() - self.arrival_time
        return self.latency_slo - elapsed


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(
            status: "RequestStatus") -> Union[FinishReason, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
