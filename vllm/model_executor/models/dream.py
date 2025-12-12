
import math
import os
import math
from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.distributed as dist

from vllm.attention.backends.abstract import AttentionType
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear, ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.configuration_dream import DreamConfig
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)
USE_VANILLA_ATTENTION = False


if os.environ.get("TRITON_INTERPRET", None) == "1":
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = True
    torch.backends.optimized_mode = False


class DreamRMSNorm(RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class DreamAttention(nn.Module):
    """Dream attention mechanism."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.layer_idx: int | None = None
        if prefix and ".layers." in prefix:
            try:
                self.layer_idx = int(prefix.split(".layers.")[-1].split(".")[0])
            except (ValueError, IndexError):
                self.layer_idx = None
        self._loggable_layers = {0, 27}
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.num_key_value_groups = max(1, self.num_heads // self.num_kv_heads)

        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        # Dream uses bias for QKV projections but not for O projection
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=True,  # Original Dream uses bias=True
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=True,  # Original Dream uses bias=True
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=True,  # Original Dream uses bias=True
            prefix=f"{prefix}.v_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,  # Original Dream uses bias=False
            prefix=f"{prefix}.o_proj"
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            attn_type=attn_type,  # CRITICAL: Pass the attention type!
            prefix=f"{prefix}.attn",
        )
        self.use_vanilla_attention = USE_VANILLA_ATTENTION

    def _log_tensor_values(self, label: str, tensor: torch.Tensor) -> None:
        return
        if tensor is None or self.layer_idx not in self._loggable_layers:
            return
        identifier = self.prefix if self.prefix else "DreamAttention"
        try:
            if tensor.ndim < 2:
                return
            seq_len = tensor.shape[0]
            if seq_len <= 8:
                return
            feat_dim = tensor.shape[-1]
            if feat_dim == 0:
                return
            slice_len = min(5, feat_dim)
            rows = []
            for idx in (6, 7, 8):
                if seq_len > idx:
                    try:
                        vals = tensor[idx, :slice_len].detach().cpu().tolist()
                    except Exception:
                        vals = "unavailable"
                    rows.append(f"pos{idx}[:{slice_len}]={vals}")
            if not rows:
                return
            print(f"[DreamAttention Debug][{identifier}] {label}: {', '.join(rows)}")
        except Exception as exc:
            print(f"[DreamAttention Debug][{identifier}] {label}: error logging values ({exc})")

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        self._log_tensor_values("after q_proj", q)
        k, _ = self.k_proj(hidden_states)
        self._log_tensor_values("after k_proj", k)
        v, _ = self.v_proj(hidden_states)
        self._log_tensor_values("after v_proj", v)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)

        self._log_tensor_values("after rope_q", q)
        self._log_tensor_values("after rope_k", k)
        # if self.layer_idx in self._loggable_layers:
        #     logger.debug(q.shape)
        #     logger.debug(k.shape)
        #     logger.debug(v.shape)
        # o = self._vanilla_attention(q, k, v)
        o = self.attn(q, k, v)
        self._log_tensor_values("after attention", o)
        if self.layer_idx in self._loggable_layers:
            identifier = self.prefix if self.prefix else "DreamAttention"
            # print(f"[DreamAttention Debug][{identifier}] attention tensor shape={tuple(o.shape)}")
        
        # 🔍 DEBUG: Log FlashAttention output  
        if hidden_states.shape[0] >= 10 and self.prefix in ["model.layers.0.self_attn", "model.layers.27.self_attn"]:
            layer_num = self.prefix.split('.')[-2]
            # o is [seq_len, num_heads * head_dim]
            # Print first 5 values of positions 6, 7, and 8
            logger.debug(f"[vLLM_FLASH] LAYER_{layer_num} pos6[:5]={o[6, :5].tolist()}")
            logger.debug(f"[vLLM_FLASH] LAYER_{layer_num} pos7[:5]={o[7, :5].tolist()}")
            logger.debug(f"[vLLM_FLASH] LAYER_{layer_num} pos8[:5]={o[8, :5].tolist()}")
        
        output, _ = self.o_proj(o)
        self._log_tensor_values("after o_proj", output)
        return output

    def _vanilla_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = q.shape[0]
        head_dim = self.head_dim
        bsz = 1

        query_states = q.view(seq_len, self.num_heads, head_dim).unsqueeze(0)
        key_states = k.view(seq_len, self.num_kv_heads, head_dim).unsqueeze(0)
        value_states = v.view(seq_len, self.num_kv_heads, head_dim).unsqueeze(0)

        query_states = query_states.permute(0, 2, 1, 3)
        key_states = key_states.permute(0, 2, 1, 3)
        value_states = value_states.permute(0, 2, 1, 3)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = torch.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.num_heads * head_dim)
        return attn_output.squeeze(0)


class DreamMLP(nn.Module):
    """Dream MLP with SiLU activation."""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        prefix : str = ""
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        x = self.act_fn(torch.cat([gate, up], dim=-1))
        x, _ = self.down_proj(x)
        return x


class DreamDecoderLayer(nn.Module):
    """Dream transformer decoder layer."""
    def __init__(
        self,
        config: DreamConfig,
        prefix: str = ""
    ) -> None:
        super().__init__()
        # Extract layer_idx from prefix like "model.layers.0"
        self.layer_idx = None
        if prefix and "layers." in prefix:
            try:
                self.layer_idx = int(prefix.split("layers.")[-1])
            except (ValueError, IndexError):
                self.layer_idx = None
        
        self.self_attn = DreamAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=True,  # Dream uses bias in attention
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = DreamMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            prefix = f"{prefix}.mlp",

        )
        self.input_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention with residual connection
        residual = hidden_states
        # 🔍 DEBUG: Log before input layernorm
        if hidden_states.shape[0] >= 10 and self.layer_idx is not None and self.layer_idx in [0, 27]:
            # hidden_states is [seq_len, hidden_size] in vLLM
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} SHAPE: hidden_states={hidden_states.shape}, positions={positions.shape}")
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} POSITIONS: {positions.tolist()}")
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} Pre_Input_Norm pos6[:5]={hidden_states[6, :5].tolist()}")
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} Pre_Input_Norm pos7[:5]={hidden_states[7, :5].tolist()}")
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} Pre_Input_Norm pos8[:5]={hidden_states[8, :5].tolist()}")
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # 🔍 DEBUG: Log after input layernorm
        if hidden_states.shape[0] >= 10 and self.layer_idx is not None and self.layer_idx in [0, 27]:
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} After_input_norm pos6[:5]={hidden_states[6, :5].tolist()}")
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} After_input_norm pos7[:5]={hidden_states[7, :5].tolist()}")
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} After_input_norm pos8[:5]={hidden_states[8, :5].tolist()}")

        hidden_states = self.self_attn(hidden_states, positions)
        
        # 🔍 DEBUG: Log after attention
        if hidden_states.shape[0] >= 10 and self.layer_idx is not None and self.layer_idx in [0, 27]:
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} After_Attn pos6[:5]={hidden_states[6, :5].tolist()}")
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} After_Attn pos7[:5]={hidden_states[7, :5].tolist()}")
            logger.debug(f"[vLLM_DECODER] Layer {self.layer_idx} After_Attn pos8[:5]={hidden_states[8, :5].tolist()}")

        hidden_states = residual + hidden_states
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, residual


class DreamModel(nn.Module):
    """Dream model for diffusion language modeling."""
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = ""
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens")
        self.layers = nn.ModuleList([
            DreamDecoderLayer(config, prefix=f"{prefix}.layers.{i}") 
            for i in range(config.num_hidden_layers)
        ])
        self.norm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 🔍 DEBUG: Log input information
        if input_ids is not None:
            logger.debug(f"[vLLM_MODEL] input_ids.shape={input_ids.shape}, input_ids={input_ids.tolist()}")
        logger.debug(f"[vLLM_MODEL] positions.shape={positions.shape}, positions={positions.tolist()}")
        
        # 🔍 DEBUG: Check embedding weights (to verify they're loaded, not random)
        logger.debug(f"[vLLM_MODEL] embed_tokens.weight.shape={self.embed_tokens.weight.shape}")
        logger.debug(f"[vLLM_MODEL] embed_tokens.weight[0, :5]={self.embed_tokens.weight[0, :5].tolist()}")
        logger.debug(f"[vLLM_MODEL] embed_tokens.weight[1, :5]={self.embed_tokens.weight[1, :5].tolist()}")
        logger.debug(f"Input_embeds: {inputs_embeds}")
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
            logger.debug(f"[vLLM_MODEL] Using inputs_embeds with shape={inputs_embeds.shape}")
        logger.debug(f"[vLLM_DECODER] Layer Pre_Input_Norm pos6[:5]={hidden_states[:, :5].tolist()}")
        for _, layer in enumerate(self.layers):
            hidden_states, _ = layer(positions, hidden_states, None)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states



class DreamModelLM(nn.Module):
    """Dream model for diffusion language modeling with LM head."""
    packed_modules_mapping = {}
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = ""
    ) -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        logger.info("Config:")
        logger.info(self.config)
        self.model = DreamModel(vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size, prefix="lm_head")
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        if getattr(self.config, 'tie_word_embeddings', False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                    sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, 
                                                   torch.Tensor]]) -> set[str]:

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head"]
                           if self.config.tie_word_embeddings else None),
        )

        loaded = loader.load_weights(weights=weights)

        return loaded