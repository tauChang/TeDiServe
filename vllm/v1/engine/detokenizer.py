# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Optional

import tokenizers
from packaging import version
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from transformers import PreTrainedTokenizerFast

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

# Only tokenizers >= 0.21.1 supports DecodeStream used for
# FastIncrementalDetokenizer.
USE_FAST_DETOKENIZER = version.parse(
    tokenizers.__version__) >= version.parse("0.21.1")

# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"


class IncrementalDetokenizer:

    def __init__(self):
        self.token_ids: dict[int, int] = {}

    @property
    def output_token_ids(self) -> list[int]:
        output_ids = []
        for i in range(len(self.token_ids)):
            output_ids.append(self.token_ids[i])
        return output_ids

    def update(self, new_token_ids: list[int],
               stop_terminated: bool) -> Optional[str]:
        self.token_ids.extend(new_token_ids)
        return None

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        return ""

    @classmethod
    def from_new_request(
        cls,
        tokenizer: Optional[AnyTokenizer],
        mask_token_id: int,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":

        assert request.sampling_params is not None

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()

        if USE_FAST_DETOKENIZER and isinstance(tokenizer,
                                               PreTrainedTokenizerFast):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, mask_token_id, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, mask_token_id, request)


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):

    def __init__(self, request: EngineCoreRequest, mask_token_id: int):
        super().__init__()

        # Stop strings
        params = request.sampling_params
        assert params is not None
        self.stop = stop = params.stop
        self.include_stop_str_in_output = params.include_stop_str_in_output

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in stop) - 1
        else:
            self.stop_buffer_length = 0
        self._last_output_text_offset: int = 0

        # Generation data
        self.prompt_length = len(request.prompt_token_ids)
        self.output_length = request.sampling_params.max_tokens
        self.mask_token_id = mask_token_id
        # Initialize all output positions with mask tokens (Dream diffusion approach)
        self.token_ids: dict[int, int] = {i: mask_token_id
                                          for i in range(self.output_length)}
        self.num_unmasked_tokens = 0
        self.output_text = ""

    def update(self, new_token_ids: list[tuple[int, int]],
               stop_terminated: bool) -> Optional[str]:
        """
        Update RequestState for the request_id by:
            1) Track newly unmasked token ids.
            2) When all tokens are unmasked, decode the entire sequence.
            3) Evaluate stop criteria.

        Return matched stop string or None.
        
        For Dream diffusion models:
        - Tokens are updated position-by-position as masks are replaced
        - We only decode once ALL mask tokens have been replaced
        - This matches the official Dream implementation behavior
        """
        if not new_token_ids:
            # Skip if no new token ids.
            return None

        # Track unmasked tokens by position
        for pos, token_id in new_token_ids:
            pos -= self.prompt_length
            if 0 <= pos < self.output_length:
                # Only count if this position was previously a mask token
                if self.token_ids[pos] == self.mask_token_id:
                    self.num_unmasked_tokens += 1
                self.token_ids[pos] = token_id
        
        # Check if all positions have been unmasked
        # For Dream: we decode ONLY when generation is complete
        if self.num_unmasked_tokens == self.output_length:
            # Call subclass-specific method to decode complete sequence
            self.output_text = self._decode_complete()
            
            # Generation complete - trigger stop
            return "stop"
        
        # Generation still in progress - no text output yet
        return None

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:
        """Decode a single token incrementally."""
        raise NotImplementedError
    
    @abstractmethod
    def _decode_complete(self) -> str:
        """Decode the complete sequence when all tokens are unmasked.
        
        Called when num_unmasked_tokens == output_length.
        Different implementations for different models:
        - FastIncrementalDetokenizer: decode one-by-one for Llada
        - SlowIncrementalDetokenizer: decode entire sequence at once for Dream
        """
        raise NotImplementedError

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """Return the output text.
        
        For Dream diffusion models:
        - Text is only available once all tokens are decoded (when update() returns "stop")
        - No incremental streaming since all tokens are generated together
        """
        # Simply return the accumulated output text
        # For Dream, this will be empty until all tokens are decoded
        return self.output_text


class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):

    def __init__(self, tokenizer: PreTrainedTokenizerFast,
                mask_token_id: int,
                request: EngineCoreRequest):
        super().__init__(request, mask_token_id)

        sampling_params = request.sampling_params
        assert sampling_params is not None

        self.request_id = request.request_id
        self.skip_special_tokens = sampling_params.skip_special_tokens
        self.stream = DecodeStream(
            skip_special_tokens=self.skip_special_tokens)

        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # Find a safe place to start.
        prompt_suffix = request.prompt_token_ids
        prompt_len = len(prompt_suffix)
        if prompt_len > 4:
            for i in range(4, min(prompt_len + 1, 24)):
                suffix = request.prompt_token_ids[-i:]
                if '�' not in self.tokenizer.decode(suffix):
                    prompt_suffix = suffix
                    break

        # Prime the stream.
        for tid in prompt_suffix:
            self._protected_step(tid)

        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens)

        if not self.spaces_between_special_tokens:
            # Store dict of added token ids so that we can suppress
            # the spaces between them.
            if (added_token_ids := getattr(self.tokenizer, "added_token_ids",
                                           None)) is None:
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content
                    for tid, tok in
                    self.tokenizer.get_added_tokens_decoder().items()
                }

            if added_token_ids:
                self.last_special = False
                self.added_token_ids = added_token_ids
            else:
                # No added tokens.
                self.spaces_between_special_tokens = True

    def decode_next(self, next_token_id: int) -> str:
        token = self._protected_step(next_token_id)

        if not self.spaces_between_special_tokens:
            special_token = self.added_token_ids.get(next_token_id)
            is_special = special_token is not None
            if is_special and self.last_special:
                # Return raw token string without any prefixed spaces.
                token = special_token
            self.last_special = is_special

        return token or ""

    def _protected_step(self, next_token_id: int) -> Optional[str]:
        try:
            token = self.stream.step(self.tokenizer, next_token_id)
        except Exception as e:
            if str(e) != INVALID_PREFIX_ERR_MSG:
                raise e
            # Recover from edge case where tokenizer can produce non-monotonic,
            # invalid UTF-8 output, which breaks the internal state of
            # tokenizers' DecodeStream.
            # See https://github.com/vllm-project/vllm/issues/17448.
            logger.warning(
                "Encountered invalid prefix detokenization error"
                " for request %s, resetting decode stream.", self.request_id)
            self.stream = DecodeStream(self.skip_special_tokens)
            token = self.stream.step(self.tokenizer, next_token_id)
        return token
    
    def _decode_complete(self) -> str:
        """Decode complete sequence token-by-token using DecodeStream.
        
        For Llada diffusion model: processes tokens one at a time through
        the DecodeStream which maintains proper streaming context.
        """
        output_text = ""
        for i in range(self.output_length):
            next_token = self.decode_next(self.token_ids[i])
            output_text += next_token
        return output_text


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):

    def __init__(self, tokenizer: AnyTokenizer, mask_token_id: int, request: EngineCoreRequest):
        super().__init__(request, mask_token_id)

        self.tokenizer = tokenizer
        params = request.sampling_params
        assert params is not None

        # Metadata for incremental detokenization.
        self.tokens, self.prefix_offset, self.read_offset = (
            convert_prompt_ids_to_tokens(
                tokenizer=tokenizer,
                prompt_ids=request.prompt_token_ids,
                skip_special_tokens=params.skip_special_tokens,
            ))

        self.prompt_len = len(request.prompt_token_ids)

        self.skip_special_tokens = params.skip_special_tokens
        self.spaces_between_special_tokens = (
            params.spaces_between_special_tokens)

    @property
    def output_token_ids(self) -> list[int]:
        return [self.token_ids[i] for i in sorted(self.token_ids.keys())]

    def decode_next(self, next_token_id: int) -> str:
        all_input_ids = [self.token_ids[i] for i in sorted(self.token_ids.keys())]

        new_tokens, decoded_text, prefix_offset, read_offset = (
            detokenize_incrementally(
                tokenizer=self.tokenizer,
                all_input_ids=all_input_ids,
                prev_tokens=self.tokens,
                prefix_offset=self.prefix_offset,
                read_offset=self.read_offset,
                skip_special_tokens=self.skip_special_tokens,
                spaces_between_special_tokens=self.
                spaces_between_special_tokens,
            ))

        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset

        return decoded_text
    
    def _decode_complete(self) -> str:
        """Decode entire sequence at once using tokenizer.decode().
        
        For Dream diffusion model: tokens are generated out-of-order by 
        confidence, so we decode the complete sequence in one call rather 
        than incrementally. This matches the official Dream implementation.
        """
        # Build complete token sequence in order
        all_output_tokens = [self.token_ids[i] for i in range(self.output_length)]
        
        # Decode entire sequence at once
        return self.tokenizer.decode(
            all_output_tokens,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )