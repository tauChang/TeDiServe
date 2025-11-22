# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import os
import gc
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.layer import Attention
from vllm.compilation.counter import compilation_counter
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config, update_config)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.distributed.parallel_state import (
    get_pp_group, get_tp_group, graph_capture, is_global_first_rank,
    prepare_communication_buffer_for_model)
from vllm.forward_context import (DPMetadata, get_forward_context,
                                  set_forward_context)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.models.interfaces_base import (VllmModelForPooling,
                                                        is_pooling_model)
from vllm.pooling_params import PoolingParams, PoolingTask
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        GiB_bytes, LazyLoader, check_use_alibi, get_dtype_size,
                        is_pin_memory_available, round_up)
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder, CommonAttentionMetadata,
    make_local_attention_virtual_batches)
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                        ChunkedLocalAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, 
                                        SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.utils import TimeProfiler

from ..sample.logits_processor import LogitsProcessorManager
from .utils import (bind_kv_cache, initialize_kv_cache_for_kv_sharing)

if TYPE_CHECKING:
    import xgrammar as xgr
    import xgrammar.kernels.apply_token_bitmask_inplace_torch_compile as xgr_torch_compile  # noqa: E501

    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    xgr_torch_compile = LazyLoader(
        "xgr_torch_compile", globals(),
        "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile")

logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        executor_id: int,
        rank: int
    ):
        self.executor_id = executor_id
        self.rank = rank # rank in this executor
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        
        profiler_path = os.path.join(
            self.vllm_config.experiment_config.experiment_dir,
            f"profiles/model_runners/{self.executor_id}_{self.rank}.jsonl"
        )
        self.time_profiler = TimeProfiler(f"model_runner_{self.executor_id}_{self.rank}",
                                                profiler_path)
        # assert the profiler path dir exist
        assert os.path.exists(os.path.dirname(profiler_path)), \
            f"Profiler path dir {os.path.dirname(profiler_path)} does not exist."

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.is_pooling_model = model_config.pooler_config is not None
        self.max_model_len = model_config.max_model_len
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size

        # Sampler
        self.sampler = Sampler()

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        self.attn_metadata_builders: list[AttentionMetadataBuilder] = []
        self.attn_backends: list[type[AttentionBackend]] = []
        # self.kv_cache_config: KVCacheConfig

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
        )

        self.use_cuda_graph = (
            self.vllm_config.compilation_config.level
            == CompilationLevel.PIECEWISE
            and self.vllm_config.compilation_config.use_cudagraph
            and not self.model_config.enforce_eager)
        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        # The convention is different.
        # self.cudagraph_batch_sizes sorts in ascending order.
        # The batch sizes in the config are in descending order.
        self.cudagraph_batch_sizes = list(
            reversed(self.compilation_config.cudagraph_capture_sizes))

        self.full_cuda_graph = self.compilation_config.full_cuda_graph

        # Cache the device properties.
        self._init_device_properties()

        # Persistent buffers for CUDA graphs.
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.query_start_loc = torch.zeros(self.max_num_reqs + 1,
                                           dtype=torch.int32,
                                           device=self.device)
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int32,
                                    device=self.device)
        self.slot_mapping = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int64,
                                        device=self.device)
        self.exec_start_pos_cpu = torch.zeros(self.max_num_reqs,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.num_exec_tokens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = check_use_alibi(model_config)

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len,
                                       self.max_num_tokens),
                                   dtype=np.int64)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        self.attn_metadata_builders[0].reorder_batch(self.input_batch,
                                                     scheduler_output)

        # For models with multiple KV cache groups, the groups should agree on
        # the same order of requests. We ensure this by only allowing the first
        # group to reorder the batch and asserting that all other groups do not
        # reorder the batch.
        # TODO(tdoublep): make this more flexible so that any group can
        # re-order the batch (not only the first).
        # TODO(tdoublep): verify this during engine init instead of at runtime
        for i in range(1, len(self.kv_cache_config.kv_cache_groups)):
            batch_reordered = self.attn_metadata_builders[i].reorder_batch(
                self.input_batch, scheduler_output)
            assert not batch_reordered

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties
        """
        self.device_properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.cuda.synchronize()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        self.input_batch.sampling_metadata_needs_refresh = False
        # gets updated to True if cur_block_start or block_size changes
        def update_and_flag(target, req_index, new_value):
            changed = target[req_index] != new_value
            target[req_index] = new_value
            return changed

        # Remove finished requests from the cached states.
        for req_id in scheduler_output.free_req_ids:
            self.requests.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where free_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.free_req_ids:
            self.input_batch.remove_request(req_id)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if pooling_params:
                assert (task := pooling_params.task) is not None, (
                    "You did not set `task` in the API")

                model = cast(VllmModelForPooling, self.model)
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                num_denoise_ran=new_req_data.num_denoise_ran,
                output_token_ids=[self.model_config.mask_token_id] * new_req_data.output_length,
                unmasked_token_ids=new_req_data.unmasked_token_ids,
                lora_request=new_req_data.lora_request,
                output_length=new_req_data.output_length,
                cur_block_start=new_req_data.cur_block_start,
                denoise_block_size=new_req_data.denoise_block_size,
            )
            for pos, token_id in new_req_data.unmasked_token_ids:
                self.requests[req_id].update_output_token_id(pos, token_id)

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        refresh = False
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            # num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            # req_state.num_computed_tokens = num_computed_tokens
            req_state.num_denoise_ran = req_data.num_denoise_ran[i]
            req_state.cur_block_start = req_data.cur_block_start[i]
            req_state.denoise_block_size = req_data.denoise_block_size[i]

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = len(new_token_ids)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.unmasked_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.unmasked_token_ids.extend(
                        new_token_ids[-num_new_tokens:])
                
                for pos, token_id in new_token_ids:
                    req_state.update_output_token_id(pos, token_id)

            # Update the block IDs.
            if not resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            # self.input_batch.num_computed_tokens_cpu[req_index] = (0) 
            if len(new_block_ids) > 0:
                logger.debug(
                    f"appending new block ids {new_block_ids} for req_index {req_index}")
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                for pos, token_id in req_data.new_token_ids[i]:
                    self.input_batch.token_ids_cpu[
                        req_index, pos] = token_id
            
            update_and_flag(self.input_batch.num_denoise_ran, req_index, req_state.num_denoise_ran)
            refresh |= update_and_flag(self.input_batch.cur_block_start_cpu, req_index, req_state.cur_block_start)
            logger.debug(f"req_index {req_index} cur_block_start updated to {req_state.cur_block_start}")
            refresh |= update_and_flag(self.input_batch.denoise_block_size_cpu, req_index, req_state.denoise_block_size)
            logger.debug(f"req_index {req_index} denoise_block_size updated to {req_state.denoise_block_size}")

        self.input_batch.sampling_metadata_needs_refresh = refresh

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[dict[str,Any], bool, torch.Tensor, list[float], np.ndarray]:
        """
        :return: tuple[
            attn_metadata: layer-to-attention_metadata mapping,
            attention_cuda_graphs: whether attention can run in cudagraph
            logits_indices, spec_decode_metadata
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        max_num_scheduled_tokens = max(tokens)
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        try:
            self.num_exec_tokens_cpu[:num_reqs].copy_(
                torch.from_numpy(num_scheduled_tokens), non_blocking=True)
        except Exception as e:
            logger.error(
                f"req_ids: {req_ids}, "
                f"num_scheduled_tokens: {num_scheduled_tokens}, "
                f"num_reqs: {num_reqs}, "
                f"total_num_scheduled_tokens: {total_num_scheduled_tokens}")
            raise e

        exec_start_pos = np.array(
            [scheduler_output.exec_start_pos[i] for i in req_ids],
            dtype=np.int32
        )
        try:
            self.exec_start_pos_cpu[:num_reqs].copy_(
                torch.from_numpy(exec_start_pos), non_blocking=True)
        except Exception as e:
            logger.error(
                f"req_ids: {req_ids}, "
                f"exec_start_pos: {exec_start_pos}, "
                f"num_scheduled_tokens: {num_scheduled_tokens}, "
                f"num_reqs: {num_reqs}, "
                f"total_num_scheduled_tokens: {total_num_scheduled_tokens}")
            raise e

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        # logger.debug(f"req_indices: {req_indices}")

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(exec_start_pos[req_indices],
               arange,
               out=positions_np)
        # logger.debug(f"exec_start_pos: {exec_start_pos}, "
        #                 f"positions_np: {positions_np}")    
        
        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])
                        
        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        # self.seq_lens_np[:num_reqs] = (
        #     self.input_batch.num_computed_tokens_cpu[:num_reqs] +
        #     num_scheduled_tokens)
        self.seq_lens_np[:num_reqs] = self.input_batch.num_tokens[:num_reqs]

        # Copy the tensors to the GPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
            
        # Common case (1D positions)
        self.positions[:total_num_scheduled_tokens].copy_(
            self.positions_cpu[:total_num_scheduled_tokens],
            non_blocking=True)

        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with 0 for full cuda graph mode.
        self.seq_lens[num_reqs:].fill_(0)
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc[num_reqs + 1:].fill_(
            self.query_start_loc_cpu[num_reqs].item())

        query_start_loc = self.query_start_loc[:num_reqs + 1]

        spec_decode_common_attn_metadata = None

        attn_metadata: dict[str, Any] = {}
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):

            blk_table = self.input_batch.block_table[kv_cache_group_id]
            blk_table_tensor = blk_table.get_device_tensor()[:num_reqs]
            slot_mapping = blk_table.slot_mapping[:total_num_scheduled_tokens]
            # logger.debug(
            #     f"in _prepare_inputs, "
            #     f"blk_tabler: {blk_table}, "
            #     f"slot_mapping: {slot_mapping}")

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode.
            blk_table.slot_mapping[total_num_scheduled_tokens:].fill_(-1)

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=self.query_start_loc[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1],
                seq_lens=self.seq_lens[:num_reqs],
                seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                num_computed_tokens_cpu=self.input_batch.
                num_computed_tokens_cpu_tensor[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                block_table_tensor=blk_table_tensor,
                slot_mapping=slot_mapping,
            )

            if self.speculative_config and \
                spec_decode_common_attn_metadata is None:
                spec_decode_common_attn_metadata = common_attn_metadata

            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          ChunkedLocalAttentionSpec):
                common_attn_metadata = make_local_attention_virtual_batches(
                    kv_cache_group_spec.kv_cache_spec.attention_chunk_size,
                    common_attn_metadata, self.cache_config.block_size)

            # Prepare for cascade attention if enabled & beneficial.
            common_prefix_len = 0
            builder = self.attn_metadata_builders[kv_cache_group_id]

            attn_metadata_i = (builder.build(
                common_prefix_len=common_prefix_len,
                common_attn_metadata=common_attn_metadata,
            ))

            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        attention_cuda_graphs = all(
            b.can_run_in_cudagraph(common_attn_metadata)
            for b in self.attn_metadata_builders)

        logits_indices = query_start_loc[1:] - 1

        confidence_thresholds = [
            scheduler_output.confidence_thresholds[i] for i in req_ids]

        return (attn_metadata, attention_cuda_graphs, logits_indices,
                confidence_thresholds,num_scheduled_tokens)

    def get_model(self) -> nn.Module:
        return self.model

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        return list(model.pooler.get_supported_tasks())

    def sync_and_slice_intermediate_tensors(
            self, num_tokens: int, intermediate_tensors: IntermediateTensors,
            sync_self: bool) -> IntermediateTensors:

        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        enabled_sp = self.compilation_config.pass_config. \
            enable_sequence_parallelism
        if enabled_sp:
            # When sequence parallelism is enabled, we always pad num_tokens
            # to be a multiple of tensor_parallel_size (tp) earlier
            assert num_tokens % tp == 0
        is_residual_scattered = tp > 1 and enabled_sp \
            and num_tokens % tp == 0

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = "residual" and is_residual_scattered
                copy_len = num_tokens // tp if is_scattered else \
                    num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True)

        return IntermediateTensors({
            k:
            v[:num_tokens // tp]
            if k == "residual" and is_residual_scattered else v[:num_tokens]
            for k, v in self.intermediate_tensors.items()
        })

    def get_dp_padding(self,
                       num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank

        # For DP: Don't pad when setting enforce_eager.
        # This lets us set enforce_eager on the prefiller in a P/D setup and
        # still use CUDA graphs (enabled by this padding) on the decoder.
        #
        # TODO(tms) : There are many cases where padding is enabled for
        # prefills, causing unnecessary and excessive padding of activations.

        if dp_size == 1 or self.vllm_config.model_config.enforce_eager:
            # Early exit.
            return 0, None

        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank)
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp).item()
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                                dp_size,
                                                device="cpu",
                                                dtype=torch.int32)
        return max_tokens_across_dp_cpu - num_tokens, num_tokens_after_padding

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        finished_sending: Optional[set[str]],
        finished_recving: Optional[set[str]],
    ) -> ModelRunnerOutput:
        assert self.input_batch.num_reqs ==\
            len(self.input_batch.pooling_params), \
        "Either all or none of the requests in" \
        " a batch must be pooling request"

        extracted_hidden_states = list(
            torch.split(hidden_states[:num_scheduled_tokens],
                        num_scheduled_tokens_np.tolist()))

        pooling_metadata = self.input_batch.pooling_metadata

        raw_pooler_output = self.model.pooler(
            hidden_states=extracted_hidden_states,
            pooling_metadata=pooling_metadata)

        pooler_output: list[Optional[torch.Tensor]] = []
        seq_lens = self.seq_lens[:self.input_batch.num_reqs]
        for raw_output, seq_len, prompt_len in zip(
                raw_pooler_output, seq_lens, pooling_metadata.prompt_lens):

            if seq_len == prompt_len:
                pooler_output.append(raw_output.data.cpu())
            else:
                pooler_output.append(None)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            finished_sending=finished_sending,
            finished_recving=finished_recving,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        with self.time_profiler.section("total"):
            try:
                logger.debug(f"start of execute_model, scheduler_output: "
                            f"{scheduler_output}, intermediate_tensors: "
                            f"{intermediate_tensors}")
                
                with self.time_profiler.section("_update_states"):
                    self._update_states(scheduler_output)
                
                    if not scheduler_output.total_num_scheduled_tokens:
                        # time_profiler total and _update_states will be
                        # overwritten next time. Do nothing
                        if not has_kv_transfer_group():
                            # Return empty ModelRunnerOutput if there's no work to do.
                            return EMPTY_MODEL_RUNNER_OUTPUT

                        return self.kv_connector_no_forward(scheduler_output)
                
                # Prepare the decoder inputs.
                with self.time_profiler.section("_prepare_inputs"):
                    (attn_metadata, attention_cuda_graphs, logits_indices,
                    confidence_thresholds,num_scheduled_tokens_np) = (
                        self._prepare_inputs(scheduler_output))

                with self.time_profiler.section("padding_and_input_prep"):
                    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
                    if (self.use_cuda_graph
                            and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
                        # Use piecewise CUDA graphs.
                        # Add padding to the batch size.
                        num_input_tokens = self.vllm_config.pad_for_cudagraph(
                            num_scheduled_tokens)
                    else:
                        # Eager mode.
                        # Pad tokens to multiple of tensor_parallel_size when
                        # enabled collective fusion for SP
                        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
                        if self.compilation_config.pass_config. \
                            enable_sequence_parallelism and tp_size > 1:
                            num_input_tokens = round_up(num_scheduled_tokens, tp_size)
                        else:
                            num_input_tokens = num_scheduled_tokens

                    # Padding for DP
                    num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
                    num_input_tokens += num_pad

                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None
                    positions = self.positions[:num_input_tokens]

                    if get_pp_group().is_first_rank:
                        intermediate_tensors = None
                    else:
                        intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                            num_input_tokens, intermediate_tensors, True)

                    # Some attention backends only support CUDA Graphs in pure decode.
                    # If attention doesn't support CUDA Graphs for this batch, but we
                    # compiled with full CUDA graphs, we have to skip them entirely.
                    skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

                with self.time_profiler.section("model_forward"):
                    # Run the model.
                    # Use persistent buffers for CUDA graphs.
                    with set_forward_context(
                            attn_metadata,
                            self.vllm_config,
                            num_tokens=num_input_tokens,
                            num_tokens_across_dp=num_tokens_across_dp,
                            skip_cuda_graphs=skip_cuda_graphs,
                    ):
                        self.maybe_setup_kv_connector(scheduler_output)

                        # logger.debug(f"Running model with input_ids: {input_ids}, "
                        #             f"positions: {positions}, "
                        #             f"intermediate_tensors: {intermediate_tensors}, ")
                        self._sync_device()
                        model_start_time = time.time()
                        model_output = self.model(
                            input_ids=input_ids,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                            inputs_embeds=inputs_embeds,
                        )
                        self._sync_device()
                        logger.debug(f"Model forward took {time.time() - model_start_time} "
                                    f"seconds")

                        self.maybe_wait_for_kv_save()
                        finished_sending, finished_recving = (
                            self.get_finished_kv_transfers(scheduler_output))

                    hidden_states = model_output

                with self.time_profiler.section("broadcast_and_logits"):
                    # to make sure we are synced across pp ranks
                    # TODO: Support overlapping mirco-batches
                    # https://github.com/vllm-project/vllm/issues/18019
                    broadcast_pp_output = \
                        self.parallel_config.distributed_executor_backend \
                        == "external_launcher" and len(get_pp_group().ranks) > 0
                    if not get_pp_group().is_last_rank:
                        logger.debug(f"PP rank {get_pp_group().rank} is not the last rank")
                        # For mid-pipeline stages, return the hidden states.
                        if not broadcast_pp_output:
                            if finished_sending or finished_recving:
                                hidden_states.finished_sending = finished_sending
                                hidden_states.finished_recving = finished_recving
                            return hidden_states
                        assert isinstance(hidden_states, IntermediateTensors)
                        logger.debug(f"PP rank {get_pp_group().rank} is sending hidden states {hidden_states}")
                        get_pp_group().send_tensor_dict(hidden_states.tensors,
                                                        all_gather_group=get_tp_group())
                        logits = None
                    else:
                        if self.input_batch.pooling_params:
                            return self._pool(hidden_states, num_scheduled_tokens,
                                            num_scheduled_tokens_np, finished_sending,
                                            finished_recving)

                        # sample_hidden_states = hidden_states[logits_indices]
                        # logits = self.model.compute_logits(sample_hidden_states, None)
                        self._sync_device()
                        logits = self.model.compute_logits(hidden_states, None)
                        self._sync_device()
                    if broadcast_pp_output:
                        model_output_broadcast_data = {
                            "logits": logits.contiguous(),
                        } if logits is not None else {}
                        model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                            model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
                        assert model_output_broadcast_data is not None
                        logits = model_output_broadcast_data["logits"]


                with self.time_profiler.section("sampling"):
                    # Sample the next token and get logprobs if needed.
                    sampling_metadata = self.input_batch.sampling_metadata
                    try:
                        sampler_output = self.sampler(
                            is_mask=input_ids == self.model_config.mask_token_id,
                            logits=logits,
                            exec_start_pos=self.exec_start_pos_cpu[:self.input_batch.num_reqs],
                            num_exec_tokens=self.num_exec_tokens_cpu[:self.input_batch.num_reqs],
                            sampling_metadata=sampling_metadata,
                            confidence_thresholds=confidence_thresholds,
                            confidences=self.input_batch.confidences,
                            req_ids=self.input_batch.req_ids,
                            req_id_to_index=self.input_batch.req_id_to_index,
                        )
                        # logger.debug(f"Sampler output: {sampler_output}")
                    except Exception as e:
                        logger.error(f"Error in sampler: {e}")
                        raise e
                    self._sync_device()

                with self.time_profiler.section("num_nans_and_logprobs"):
                    num_nans_in_logits = {}
                    if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
                        num_nans_in_logits = self._get_nans_in_logits(logits)

                    # # TODO(woosuk): The following loop can be slow since it iterates over
                    # # the requests one by one. Optimize.
                    # loop_start_time = time.time()
                    # discard_sampled_tokens_req_indices = []
                    # for i, req_id in enumerate(self.input_batch.req_ids):
                    #     req_state = self.requests[req_id]
                    #     seq_len = (req_state.num_computed_tokens +
                    #                scheduler_output.num_scheduled_tokens[req_id])
                    #     if seq_len < req_state.num_tokens:
                    #         # Ignore the sampled token for partial prefills.
                    #         # Rewind the generator state as if the token was not sampled.
                    #         # This relies on cuda-specific torch-internal impl details
                    #         generator = self.input_batch.generators.get(i)
                    #         if generator is not None:
                    #             generator.set_offset(generator.get_offset() - 4)
                    #         # Record the index of the request that should not be sampled,
                    #         # so that we could clear the sampled tokens before returning.
                    #         discard_sampled_tokens_req_indices.append(i)
                    # logger.debug(f"discard_sampled_tokens: {discard_sampled_tokens_req_indices}")
                    # logger.debug(f"Loop over requests took "
                    #             f"{time.time() - loop_start_time} seconds")

                    # NOTE: GPU -> CPU Sync happens here.
                    # Move as many CPU operations as possible before this sync point.
                    logprobs_tensors = sampler_output.logprobs_tensors
                    logprobs_lists = logprobs_tensors.tolists() \
                        if logprobs_tensors is not None else None

                    # Compute prompt logprobs if needed.
                    prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                        hidden_states[:num_scheduled_tokens],
                        scheduler_output,
                    )

                with self.time_profiler.section("valid_token_ids"):
                    # [tau_chang]: Get the valid generated tokens.
                    valid_sampled_token_ids = sampler_output.sampled_token_ids
                    logger.debug(f"Valid sampled token ids: {valid_sampled_token_ids}")

                    # Mask out the sampled tokens that should not be sampled.
                    # for i in discard_sampled_tokens_req_indices:
                    #     valid_sampled_token_ids[i].clear()

                    # Cache the sampled tokens in the model runner, so that the scheduler
                    # doesn't need to send them back.
                    # NOTE(woosuk): As an exception, when using PP, the scheduler sends
                    # the sampled tokens back, because there's no direct communication
                    # between the first-stage worker and the last-stage worker.
                    for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
                        if not sampled_ids:
                            continue
                    
                        for pos, token_id in sampled_ids:
                            self.input_batch.token_ids_cpu[req_idx, pos] = token_id

                            req_id = self.input_batch.req_ids[req_idx]
                            req_state = self.requests[req_id]
                            req_state.unmasked_token_ids.append((pos, token_id))
                            req_state.update_output_token_id(pos, token_id)

                        # start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                        # end_idx = start_idx + len(sampled_ids)
                        # assert end_idx <= self.max_model_len, (
                        #     "Sampled token IDs exceed the max model length. "
                        #     f"Total number of tokens: {end_idx} > max_model_len: "
                        #     f"{self.max_model_len}")

                        # self.input_batch.token_ids_cpu[req_idx,
                        #                                start_idx:end_idx] = sampled_ids
                        # self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                        # self.input_batch.num_tokens[req_idx] = end_idx
                        # req_id = self.input_batch.req_ids[req_idx]
                        # req_state = self.requests[req_id]
                        # req_state.output_token_ids.extend(sampled_ids)

                    self._sync_device()
                
                with self.time_profiler.section("build_output"):
                    model_runner_output = ModelRunnerOutput(
                        req_ids=self.input_batch.req_ids,
                        req_id_to_index=self.input_batch.req_id_to_index,
                        sampled_token_ids=valid_sampled_token_ids,
                        confidence_stats=sampler_output.confidence_stats,
                        spec_token_ids=None,
                        logprobs=logprobs_lists,
                        prompt_logprobs_dict=prompt_logprobs_dict,
                        pooler_output=[],
                        finished_sending=finished_sending,
                        finished_recving=finished_recving,
                        num_nans_in_logits=num_nans_in_logits,
                    )
                
            except Exception as e:
                # write to file this error
                import traceback
                logger.error(f"Exception in execute_model: {e}")
                logger.error(traceback.format_exc())
                raise e
        
        self.time_profiler.add_info(f"num_input_tokens", num_input_tokens)
        self.time_profiler.commit()
        return model_runner_output

    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(
                scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            kv_connector.start_load_kv(get_forward_context())

    @staticmethod
    def maybe_wait_for_kv_save() -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    @staticmethod
    def get_finished_kv_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.free_req_ids)
        return None, None

    def kv_connector_no_forward(
            self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        with set_forward_context(None, self.vllm_config):
            self.maybe_setup_kv_connector(scheduler_output)
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))

        if not finished_sending and not finished_recving:
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.finished_sending = finished_sending
        output.finished_recving = finished_recving
        return output

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, \
                f"Config `{config_name}` not supported. " \
                f"Allowed configs: {allowed_config_names}"
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            time_before_load = time.perf_counter()
            model_loader = get_model_loader(self.load_config)
            if not hasattr(self, "model"):
                logger.info("Loading model from scratch...")
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config,
                    model_config=self.model_config)
            else:
                logger.info(
                    "Model was already initialized. Loading weights inplace..."
                )
                model_loader.load_weights(self.model,
                                          model_config=self.model_config)
            time_after_load = time.perf_counter()
        self.model_memory_usage = m.consumed_memory
        logger.info("Model loading took %.4f GiB and %.6f seconds",
                    self.model_memory_usage / GiB_bytes,
                    time_after_load - time_before_load)
        prepare_communication_buffer_for_model(self.model)


    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():

            num_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True)

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1)
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc_np[req_idx].item()
            prompt_hidden_states = hidden_states[offset:offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states, None)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs,
                                                         non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: Optional[torch.Tensor],
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None
                    and req_index < logits.shape[0] else 0)
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(self, input_ids: torch.Tensor):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run 
        """
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        # randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        randomize_inputs = False
        if not randomize_inputs:
            yield
        else:
            import functools

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                    dtype=input_ids.dtype)

            logger.debug("Randomizing dummy data for DP Rank")
            input_ids.copy_(rand_input_ids()[:input_ids.size(0)],
                            non_blocking=True)
            yield
            input_ids.fill_(0)

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        capture_attn_cudagraph: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
        use_attn_metadata: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        logger.debug(f"Dummy run with num_tokens: {num_tokens}, "
                     f"num_pad: {num_pad}, ")
        num_tokens += num_pad

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        # num_reqs = min(num_tokens, max_num_reqs)
        # if (self.model_config.cache_prefix and \
        #     self.model_config.cache_suffix) and \
        #         self.model_config.denoise_block_size > 0:
        #     seq_len = self.model_config.denoise_block_size
        # else:
        #     # seq_len = min(num_tokens, 512)
        #     # has to use num_tokens
        #     seq_len = min(num_tokens, self.max_model_len)

        # With more num_reqs, the attention computation becomes cheaper.
        # So here we use the worst case seq_len (least num_reqs) to profile
        # the worst case latency.
        seq_len = min(num_tokens, self.max_model_len)
        # num_reqs = max(1, num_tokens // seq_len)
        num_reqs = max(1, -(-num_tokens // seq_len))  # Ceiling division
        logger.debug(f"Dummy run with num_reqs: {num_reqs} and seq_len: {seq_len}")
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        attn_metadata: Optional[dict[str, Any]] = None
        # if capture_attn_cudagraph:
        if use_attn_metadata:
            attn_metadata = {}

            # Make sure max_model_len is used at the graph capture time.
            # self.seq_lens_np[:num_reqs] = self.max_model_len
            self.seq_lens_np[:num_reqs] = seq_len
            self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                           non_blocking=True)

            cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)
            
            logger.debug(f"cu_num_tokens: {cu_num_tokens}")
            self.query_start_loc_np[0] = 0
            self.query_start_loc_np[1:num_reqs +1] = cu_num_tokens
            self.query_start_loc[:num_reqs + 1].copy_(
                self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
            self.query_start_loc[num_reqs +1:].fill_(
                self.query_start_loc_cpu[num_reqs].item())
            logger.debug(f"query_start_loc: {self.query_start_loc[:num_reqs+1]}")
            logger.debug(f"seq_lens: {self.seq_lens[:num_reqs]}")

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs +
                                                                 1],
                    seq_lens=self.seq_lens[:num_reqs],
                    seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=seq_len,
                    block_table_tensor=self.input_batch.block_table[
                        kv_cache_group_id].get_device_tensor()[:num_reqs],
                    slot_mapping=self.input_batch.
                    block_table[kv_cache_group_id].slot_mapping[:num_tokens])

                attn_metadata_i = self.attn_metadata_builders[
                    kv_cache_group_id].build_for_cudagraph_capture(
                        common_attn_metadata)
                for layer_name in kv_cache_group_spec.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            model = self.model
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = None
            positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device))

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False)
            

            with self.maybe_randomize_inputs(input_ids), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp):
                self._sync_device()
                start_time = time.time()
                logger.debug(f"input_ids shape: {input_ids.shape}")
                outputs = model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )
                logger.debug(f"outputs shape: {outputs.shape}")
                self._sync_device()
                logger.debug(f"in _dummy_run model forward took "
                             f"{time.time() - start_time} seconds")
            hidden_states = outputs

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states, hidden_states[logit_indices]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
        unmask_all: bool = False
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.
        seq_len = hidden_states.shape[0]
        # hidden_states = torch.rand_like(hidden_states)

        self._sync_device()
        start_time = time.time()
        logits = self.model.compute_logits(hidden_states, None)
        self._sync_device()
        logger.debug(f"in _dummy_sampler_run compute_logits took "
                        f"{time.time() - start_time} seconds")
        logger.debug(f"hidden_states shape: {hidden_states.shape}, "
                     f"logits shape: {logits.shape}")
        # num_reqs = logits.size(0)
        num_reqs = (seq_len + self.max_model_len - 1)//self.max_model_len
        chunk_lengths = []
        for i in range(num_reqs):
            if i == num_reqs -1:
                chunk_lengths.append(seq_len - i * self.max_model_len)
            else:
                chunk_lengths.append(self.max_model_len)
        
        exec_start_pos = torch.zeros((num_reqs,), dtype=torch.int32)
        num_exec_tokens = torch.tensor(chunk_lengths, dtype=torch.int32)

        dummy_tensors = lambda v: torch.full(
            (num_reqs, ), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessorManager(),
            num_prompt_tokens=[0] * num_reqs,
            num_tokens=[num_exec_tokens[i].item() for i in range(num_reqs)],
            num_denoise_ran=[0] * num_reqs,
            cur_block_start=[0] * num_reqs,
            denoise_block_size=[seq_len] * num_reqs,
        )
        if unmask_all:
            confidence_thresholds = [0.0] * num_reqs # unmask all tokens
        else:
            confidence_thresholds = [0.9] * num_reqs # usual case: only a few tokens are unmasked
        try:
            # [tau_chang] is_mask is all True
            is_mask = torch.ones(
                (seq_len), dtype=torch.bool, device=self.device)
            self._sync_device()
            start_time = time.time()
            sampler_output = self.sampler(is_mask=is_mask,
                                          logits=logits,
                                          exec_start_pos=exec_start_pos,
                                          num_exec_tokens=num_exec_tokens,
                                          sampling_metadata=dummy_metadata,
                                          confidence_thresholds=confidence_thresholds,
                                          confidences=self.input_batch.confidences,
                                          req_ids=[f"dummy_req_{i}" for i in range(num_reqs)],
                                          req_id_to_index={f"dummy_req_{i}": i for i in range(num_reqs)}
                                          )
                                          
            self._sync_device()
            logger.debug(f"in _dummy_sampler_run sampler took "
                         f"{time.time() - start_time} seconds")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e
        return sampler_output

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        hidden_states_list = list(
            torch.split(hidden_states, num_scheduled_tokens_list))

        req_num_tokens = num_tokens // num_reqs

        model = cast(VllmModelForPooling, self.model)
        dummy_task = self.get_supported_pooling_tasks()[0]
        dummy_pooling_params = PoolingParams(task=dummy_task)

        to_update = model.pooler.get_pooling_updates(dummy_task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=torch.tensor([h.shape[0] for h in hidden_states_list],
                                     device=self.device),
            prompt_token_ids=torch.zeros((num_reqs, req_num_tokens),
                                         dtype=torch.int32,
                                         device=self.device),
            pooling_params=[dummy_pooling_params] * num_reqs)

        try:
            pooler_output = model.pooler(hidden_states=hidden_states_list,
                                         pooling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e
        return pooler_output

    def memory_profile_run(self) -> None:
        # return

        # Add `is_profile` here to pre-allocate communication buffers
        logger.debug(f"max_num_tokens: {self.max_num_tokens}")
        hidden_states, last_hidden_states \
            = self._dummy_run(self.max_num_tokens, is_profile=True)
        logger.debug(f"last_hidden_states: {last_hidden_states.shape}")
        logger.debug(f"hidden_states: {hidden_states.shape}")
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                # output = self._dummy_sampler_run(last_hidden_states)
                output = self._dummy_sampler_run(hidden_states, unmask_all=True)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        gc.collect()
    
    def latency_profile_run(self, num_tokens, num_run=3, num_warmup_run=2) -> \
        list[float]:
        # Add `is_profile` here to pre-allocate communication buffers
        results = []
        for i in range(num_warmup_run + num_run):
            self._sync_device()
            start_time = time.time()
            hidden_states, last_hidden_states \
                = self._dummy_run(num_tokens, is_profile=False, use_attn_metadata=True)
            if get_pp_group().is_last_rank:
                if self.is_pooling_model:
                    output = self._dummy_pooler_run(hidden_states)
                else:
                    # output = self._dummy_sampler_run(last_hidden_states)
                    output = self._dummy_sampler_run(hidden_states)
            else:
                output = None
            self._sync_device()
            end_time = time.time()
            logger.debug(f"Latency profile run {i} for {num_tokens} tokens took {end_time - start_time:.4f} seconds")
            if i >= num_warmup_run:
                results.append(end_time - start_time)
            
        gc.collect()
        return results
        

    def capture_model(self) -> None:
        # return

        if not self.use_cuda_graph:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "set -O %s and ensure `use_cudagraph` was not manually set to "
                "False", CompilationLevel.PIECEWISE)
            return

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with graph_capture(device=self.device):
            full_cg = self.full_cuda_graph
            # Only rank 0 should print progress bar during capture
            compilation_cases = reversed(self.cudagraph_batch_sizes)
            if is_global_first_rank():
                compilation_cases = tqdm(
                    list(compilation_cases),
                    disable=not self.load_config.use_tqdm_on_load,
                    desc="Capturing CUDA graph shapes")
            for num_tokens in compilation_cases:
                # We skip EPLB here since we don't want to record dummy metrics
                for _ in range(
                        self.compilation_config.cudagraph_num_of_warmups):
                    iter_start_time = time.perf_counter()
                    self._dummy_run(num_tokens,
                                    capture_attn_cudagraph=full_cg,
                                    skip_eplb=True)
                    iter_end_time = time.perf_counter()
                    elasped_time_in_ms = (iter_end_time - iter_start_time) * 1000
                    logger.debug(
                        "Warmup iteration for capturing graph with %d tokens took %.2f ms",
                        num_tokens, elasped_time_in_ms)
                self._dummy_run(num_tokens,
                                capture_attn_cudagraph=full_cg,
                                skip_eplb=True)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_backends) == 0 and len(
            self.attn_metadata_builders
        ) == 0, "Attention backends are already initialized"
        for i, kv_cache_group_spec in enumerate(
                kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            if isinstance(kv_cache_spec, AttentionSpec):
                attn_backend_i = get_attn_backend(
                    kv_cache_spec.head_size,
                    self.dtype,
                    kv_cache_spec.dtype,
                    kv_cache_spec.block_size,
                    self.model_config.is_attention_free,
                    use_mla=kv_cache_spec.use_mla,
                )
                if attn_backend_i is None:
                    error_msg = (f"Error with get_attn_backend: "
                                 f"{kv_cache_spec.head_size=}, "
                                 f"{self.dtype=}, {kv_cache_spec.dtype=}, "
                                 f"{kv_cache_spec.block_size=}, "
                                 f"{self.model_config.is_attention_free=}, "
                                 f"{kv_cache_spec.use_mla=}")
                    logger.error(error_msg)
                    raise NotImplementedError(
                        "Non-Attention backend is not supported by V1 "
                        "GPUModelRunner.")
            else:
                raise ValueError(
                    f"Unknown KV cache spec type: {type(kv_cache_spec)}")

            attn_metadata_builder_i = attn_backend_i.get_builder_cls()(
                kv_cache_spec,
                self.vllm_config,
                self.device,
            )

            if (self.full_cuda_graph
                    and not attn_metadata_builder_i.full_cudagraph_supported):
                raise ValueError(
                    f"Full CUDAGraph not supported for "
                    f"{attn_backend_i.__name__}. Turn off CompilationConfig."
                    f"full_cuda_graph or use a different attention backend.")

            self.attn_backends.append(attn_backend_i)
            self.attn_metadata_builders.append(attn_metadata_builder_i)

    def may_reinitialize_input_batch(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]
        if block_sizes != [self.cache_config.block_size]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
            )

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
         """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(kv_cache_tensor.size,
                                 dtype=torch.int8,
                                 device=self.device)
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            layer_names.update(group.layer_names)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"
        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
            correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for i, kv_cache_group_spec in enumerate(
                kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            for layer_name in kv_cache_group_spec.layer_names:
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = self.attn_backends[
                            i].get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(
                            kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(
                            range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(kv_cache_shape[i]
                                           for i in kv_cache_stride_order)
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = kv_cache_raw_tensors[
                        layer_name].view(dtype).view(kv_cache_shape).permute(
                            *inv_order)
                else:
                    raise NotImplementedError

        return kv_caches

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                   kv_cache_raw_tensors)

        # Setup `kv_cache_config` and `kv_caches` for models
        # with cross-layer KV sharing
        if self.shared_kv_cache_layers:
            initialize_kv_cache_for_kv_sharing(
                self.shared_kv_cache_layers,
                kv_cache_config.kv_cache_groups,
                kv_caches,
            )

        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
                      self.kv_caches)
        return kv_caches

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config
        self.may_reinitialize_input_batch(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if (kv_tgt_layer :=
                    attn_module.kv_sharing_target_layer_name) is not None:
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue

            # TODO: Support other attention modules, e.g., cross-attention
            if attn_module.attn_type == AttentionType.DECODER:
                use_local_attention = (self.attention_chunk_size is not None
                                       and getattr(attn_module.impl,
                                                   "use_irope", False))
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=use_mla)
                    assert not use_local_attention, (
                        "attention module can not be with ",
                        "both local attention and sliding window")
                elif use_local_attention:
                    kv_cache_spec[layer_name] = (ChunkedLocalAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        attention_chunk_size=self.attention_chunk_size,
                        use_mla=use_mla))
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        use_mla=use_mla)
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec
