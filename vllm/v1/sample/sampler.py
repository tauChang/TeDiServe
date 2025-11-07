# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate

from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5

logger = init_logger(__name__)

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()
        self.pin_memory = is_pin_memory_available()

    def forward(
        self,
        is_mask: torch.Tensor, # [tau_chang]: shape [sum of request.num_tokens]
        logits: torch.Tensor, # [tau_chang]: shape [sum of request.num_tokens, 
                              # vocab_size]
        exec_start_pos: torch.Tensor, # [tau_chang]: shape [num_reqs]
        num_exec_tokens: torch.Tensor, # [tau_chang]: shape [num_reqs]
        sampling_metadata: SamplingMetadata,
        confidence_thresholds: list[float]
    ) -> SamplerOutput:
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        # TODO(rob): provide option for logprobs post sampling.
        # See https://vllm-dev.slack.com/archives/C07UUL8E61Z/p1735907856007919 # noqa: E501

        # [tau_chang] Ignore for now.
        # num_logprobs = sampling_metadata.max_num_logprobs
        # if num_logprobs is not None:
        #     raw_logprobs = self.compute_logprobs(logits)
        num_logprobs = None

        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        # [tau_chang] Ignore for now
        # # Apply allowed token ids.
        # logits = self.apply_allowed_token_ids(logits, sampling_metadata)
        # # Apply bad words exclusion.
        # logits = self.apply_bad_words(logits, sampling_metadata)

        # # Apply logits processors which can impact greedy sampling
        # for processor in (sampling_metadata.logitsprocs.non_argmax_invariant):
        #     logits = processor.apply(logits)

        # # Apply penalties (e.g., min_tokens, freq_penalties).
        # logits = self.apply_penalties(logits, sampling_metadata)
        # Sample the next token.

        # sampled = self.sample(logits, sampling_metadata) # [TODO (tau_chang)]: output should be List[List[Tuple[int, int]]]
        unmasked, avg_output_confidences = self.unmask(is_mask, logits, exec_start_pos, num_exec_tokens,
                               sampling_metadata, confidence_thresholds)
        logprobs_tensors = None

        # [tau_chang] Ignore for now.
        # # Convert sampled token ids to int64 (long) type to ensure compatibility
        # # with subsequent operations that may use these values as indices.
        # # This conversion is necessary because FlashInfer sampling operations
        # # return int32 (while PyTorch argmax and topk return int64).
        # sampled = sampled.long()

        # # Gather the logprobs of the topk and sampled token (if requested).
        # # Get logprobs and rank tensors (if requested)
        # logprobs_tensors = None if num_logprobs is None else \
        #     self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)

        # # Use int32 to reduce the tensor size.
        # sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            sampled_token_ids=unmasked,
            avg_output_confidences=avg_output_confidences,
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)
    
    def get_output_range(self, sampling_metadata: SamplingMetadata,
                         exec_start_pos: torch.Tensor,
                         num_exec_tokens: torch.Tensor) -> \
        list[tuple[int, int, int]]:
        # req_start is the start position of the current request relative to
        # the entire batch.
        # exec_start is the position of the first token in execution in the
        # request.
        # block_start is the position of the first token in the denoising
        # block in the request.
        req_start = 0
        ranges = []
        for i in range(len(sampling_metadata.num_tokens)):
            exec_start = exec_start_pos[i].item()
            block_start = sampling_metadata.cur_block_start[i]
            block_end = block_start + sampling_metadata.denoise_block_size[i]
            logger.debug(f"Request {i}:"
                         f"block_start={block_start}, block_end={block_end}, "
                         )
            unmask_range = (req_start + block_start - exec_start,
                            req_start + block_end - exec_start)

            is_recompute = sampling_metadata.num_tokens[i] == num_exec_tokens[i] 
            if is_recompute:
                output_confidence_range = (req_start + sampling_metadata.num_prompt_tokens[i],
                                           req_start + sampling_metadata.num_tokens[i])
            else:
                output_confidence_range = (req_start,
                                           req_start + num_exec_tokens[i].item())
            
            ranges.append(
                {
                    "unmask_range": unmask_range,
                    "output_confidence_range": output_confidence_range,
                    "block_start": block_start
                }
            )
            logger.debug(f"ranges: {ranges[-1]}")
            req_start += num_exec_tokens[i].item()
        return ranges

    
    def add_noise_to_logits(self, 
        logits: torch.Tensor, 
        sampling_metadata: SamplingMetadata) -> torch.Tensor:
        #[TODO (tau_chang)]: fix this
        return logits

        noise = torch.rand_like(logits, dtype=torch.float32)
        # gumbel_noise = (-torch.log(noise)) ** temperature
        
    def unmask(
        self,
        is_mask: torch.Tensor,
        logits: torch.Tensor,
        exec_start_pos: torch.Tensor,
        num_exec_tokens: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        confidence_thresholds: list[float]
    ) -> list[list[tuple[int, int]]]:
        """Unmask the logits based on the is_mask tensor."""
        # [tau_chang]: For now, we assume that is_mask is same shape as logits.
        logger.debug(
            f"Unmasking logits with shape {logits.shape} and is_mask with shape {is_mask.shape}"
        )
        assert logits.shape[0] == is_mask.shape[0]

        # [tau_chang]: Run operations on everything. Maybe optimize to
        # run operations only on output (not on prompt).
        
        logits_with_noise = self.add_noise_to_logits(logits, sampling_metadata)
        x_0 = torch.argmax(logits_with_noise, dim=-1)
        # logger.debug(f"x_0: {x_0}")
        p = F.softmax(logits_with_noise, dim=-1)
        confidence = p.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)

        x_0 = x_0.cpu()
        
        unmasked_tokens = []
        avg_output_confidences = []
        # avg_masked_confidence = []
        # for i, (start, end, block_start) in \
        #     enumerate(self.get_output_range(sampling_metadata, exec_start_pos, num_exec_tokens)):
        for i, ranges in \
            enumerate(self.get_output_range(sampling_metadata, exec_start_pos, num_exec_tokens)):
            start, end = ranges["unmask_range"]
            block_start = ranges["block_start"]
            output_start, output_end = ranges["output_confidence_range"]

            logger.debug(
                f"Processing range {start}:{end}, block_start={block_start}")
            x_0_slice = x_0[start:end]
            confidence_slice = confidence[start:end]
            is_mask_slice = is_mask[start:end]

            # logger.debug(f"confidence slice: {confidence_slice}")
            # logger.debug(f"is mask slice: {is_mask_slice}")
            
            assert is_mask_slice.any(), f"No masked tokens in range {start}:{end}"


            masked_confidences = confidence_slice[is_mask_slice]
            masked_indices = torch.nonzero(is_mask_slice, as_tuple=False).squeeze(1)
            # logger.debug(
            #     f"Masked indices: {masked_indices}, "
            #     f"Masked confidences: {masked_confidences}")

            selected_mask = \
                masked_confidences > confidence_thresholds[i]
            selected_indices = masked_indices[selected_mask].cpu()

            # [tau_chang]: Note that selected_indices are relative to the 
            # output range, not including the prompt.
            if selected_indices.numel() == 0:
                top_index = masked_indices[masked_confidences.argmax()]
                selected_indices = torch.tensor([top_index], dtype=torch.int32)
                logger.debug(
                    f"No masked tokens with confidence > {confidence_thresholds[i]}, selecting top token: {selected_indices}")
            # else:
            #     logger.debug(
            #         f"Token with confidence > {confidence_thresholds[i]}! Selected indices: {selected_indices}")
                
            selected_tokens = x_0_slice[selected_indices]
            # logger.debug(
            #     f"Selected tokens: {selected_tokens}, "
            #     f"Selected indices: {selected_indices}, "
            # )
            # [tau_chang]: Convert position back to absolute (including prompt).
            # append a list of tuples (index, token) to unmasked_tokens
            unmasked_tokens.append([
                (index.item() + block_start, token.item())
                for index, token in zip(selected_indices, selected_tokens)
            ])

            avg_output_confidences.append(
                confidence[output_start:output_end].mean().item()
            )
            logger.debug(
                f"Avg output confidence for range {output_start}:{output_end}: "
                f"{avg_output_confidences[-1]}"
            )

            # logger.debug(
            #     f"Unmasked tokens for range {start}:{end}: {unmasked_tokens[-1]}"
            # )
        
        return unmasked_tokens, avg_output_confidences

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """
        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and/or top_p.
        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits

    def apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask,
                                float("-inf"))
        return logits

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.bad_words_token_ids:
            apply_bad_words(
                logits,
                sampling_metadata.bad_words_token_ids,
                sampling_metadata.output_token_ids,
            )
        return logits
