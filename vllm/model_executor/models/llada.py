from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from abc import abstractmethod
from enum import Enum
from collections.abc import Iterable
from typing import Any, Optional, Union, List, cast
import warnings

from .configuration_llada import LLaDAConfig, ModelConfig

from vllm.attention import Attention, AttentionType
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata

from .interfaces import SupportsPP
from .utils import (AutoWeightsLoader, 
                    WeightsMapper,
                    is_pp_missing_parameter,
                    maybe_prefix,
                    make_layers,
                    make_empty_intermediate_tensors_factory)
                    

class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        # return RMSLayerNorm(config, size=size, **kwargs)
        return RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                print(f"weight device: {self.weight.device}, x device: {x.device}")
                return self.weight * x
        else:
            return x


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        return cast(Activation, SiLU(inplace=False))

class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class LLaDAAttention(nn.Module):
    def __init__(
        self,
        config: LLaDAConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float,
        o_bias: bool = False,
        qkv_bias: bool = False,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = config.max_sequence_length

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=o_bias,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb()
        
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )
            
    def _init_rotary_emb(self, 
                        #  config: LlamaConfig,
                        #  rope_scaling: Optional[dict[str, Any]],
                        #  quant_config: Optional[QuantizationConfig]
                         ) -> None:
        # is_neox_style = True
        # is_gguf = quant_config and quant_config.get_name() == "gguf"
        # if is_gguf and config.model_type == "llama":
        #     is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            # rope_scaling=rope_scaling,
            # is_neox_style=is_neox_style,
            # partial_rotary_factor=self.partial_rotary_factor,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LLaDATransformerLayer(nn.Module):
    def __init__(self, config: LLaDAConfig, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.d_model
        rope_theta = config.rope_theta
        bias = config.include_bias # for ff_proj
        qkv_bias = config.include_qkv_bias # for ${q, k, v}_proj

        self.attn_norm = LayerNorm.build(config)

        self.self_attn = LLaDAAttention(
            config=config,
            prefix=f"{prefix}.self_attn",
            hidden_size=self.hidden_size,
            num_heads=config.n_heads,
            num_kv_heads=config.n_kv_heads,
            rope_theta=rope_theta,
            o_bias=bias,
            qkv_bias=qkv_bias
        )

        self.ff_norm = LayerNorm.build(config)
        
        self.ff_proj = ColumnParallelLinear(
            config.d_model,
            config.mlp_hidden_size,
            bias=bias,
            prefix=f"{prefix}.ff_proj",
        )

        self.up_proj = ColumnParallelLinear(
            config.d_model,
            config.mlp_hidden_size,
            bias=bias,
            prefix=f"{prefix}.up_proj",
        )

        self.ff_out = RowParallelLinear(
            input_size=config.mlp_hidden_size,
            output_size=config.d_model,
            bias=bias,
            reduce_results=True,
            prefix=f"{prefix}.ff_out",
        )

        self.act = Activation.build(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.self_attn(positions=positions, 
                                       hidden_states=hidden_states)

        # FC
        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        (hidden_states, _), (hidden_states_up, _) = self.ff_proj(hidden_states), \
            self.up_proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * hidden_states_up
        hidden_states, _ = self.ff_out(hidden_states)
        hidden_states += residual
        
        return hidden_states
        

        
        

        
        

        

class LLaDATransformer(nn.Module):
    def __init__(self, config: LLaDAConfig, prefix: str = ""):
        super().__init__()

        assert not config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert config.rope, "Rope must be used in Llama-Encoder for MDM."
        
        self.config = config

        self.wte = VocabParallelEmbedding(
            config.embedding_size or config.vocab_size, 
            config.d_model
        )
        
        self.ln_f = LayerNorm.build(config)
        # self.blocks = nn.ModuleList([
        #     LLaDABlock.build(i, config) for i in range(config.n_layers)])

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.n_layers,
            lambda prefix: LLaDATransformerLayer(config, prefix=prefix),
            prefix=f"{prefix}.blocks",
        )
        
        if not (self.config.alibi or self.config.rope): # should be False, since RoPE is always used
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, 
                                     device=config.init_device)}
            )
        
        # if not config.weight_tying: # should be True, since weight tying is False
        #     self.ff_out = ReplicatedLinear(
        #         config.d_model,
        #         config.embedding_size or config.vocab_size,
        #         bias=config.include_bias,
        #     )
        
        if self.config.alibi: # should be False
            raise NotImplementedError("ALiBi is not implemented yet.")
            # get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            # self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # attention_bias: Optional[torch.Tensor] = None,
        # past_key_values: Optional[List[torch.Tensor]] = None,
        # use_cache: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if get_pp_group().is_first_rank:
            if input_embeds is None:
                hidden_states = self.get_input_embeddings(input_ids)
            else:
                hidden_states = input_embeds

            if not (self.config.alibi or self.config.rope): # should be False
                pos_embeds = self.transformer.wpe(positions)
                hidden_states += pos_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors.hidden_states
            
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states = layer(
                hidden_states=hidden_states,
                positions=positions,
            )
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        
        hidden_states = self.ln_f(hidden_states)

        return hidden_states
        
        # if self.config.weight_tying:
        #     logits = F.linear(hidden_states, self.wte.weight, None)
        # else:
        #     logits = self.ff_out(hidden_states)

        # return logits




class LLaDAModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        
        if config.alibi and config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")

        if config.alibi and config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")

        if config.embedding_size is not None and config.embedding_size != config.vocab_size:
            if config.embedding_size < config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif config.embedding_size % 128 != 0:

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )
        
        self.transformer = LLaDATransformer(config, prefix=f"{prefix}.transformer")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size)
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.get_input_embeddings(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # attention_bias: Optional[torch.Tensor] = None,
        # past_key_values: Optional[List[torch.Tensor]] = None,
        # use_cache: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.transformer(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            input_embeds=input_embeds,
            # attention_mask=attention_mask,
            # attention_bias=attention_bias,
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            # output_hidden_states=output_hidden_states,
        )
    
    def load_weights(self, weights: Iterable[tuple[str, 
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name) # q_proj -> qkv_proj
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LLaDAModelLM(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.model = LLaDAModel(vllm_config=vllm_config, 
                                prefix=maybe_prefix(prefix, "model"))

        if self.config.weight_tying: # should be False
            self.lm_head = self.model.transformer.wte
        else:
            self.lm_head = ParallelLMHead(config.embedding_size or config.vocab_size,
                                          config.d_model)

        self.logits_processor = LogitsProcessor(config.vocab_size)
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # attention_bias: Optional[torch.Tensor] = None,
        # past_key_values: Optional[List[torch.Tensor]] = None,
        # labels: Optional[torch.Tensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
        **kwargs
    ):
        # if use_cache is None:
        #     use_cache = self.config.use_cache
        
        # if output_attentions:
        #     raise ValueError("LLaDA does not support output_attentions.")

        # return_dict = return_dict if return_dict is not None else self.config.model_config.hf_config.use_return_dict

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            input_embeds=input_embeds,
            # attention_mask=attention_mask,
            # attention_bias=attention_bias,
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            # output_hidden_states=output_hidden_states,
        )
        
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                    sampling_metadata)

        return logits

    def load_weights(self, weights: Iterable[tuple[str, 
                                                   torch.Tensor]]) -> set[str]:
        # print(f"all weights name: {[name for name, _ in weights]}")
        # print(f"all model parameters: {[name for name, _ in self.named_parameters(remove_duplicate=False)]}")
    
        mapper = WeightsMapper(
            orig_to_new_substr={
                ".blocks.": ".layers.",
                ".q_proj.": ".self_attn.q_proj.",
                ".k_proj.": ".self_attn.k_proj.",
                ".v_proj.": ".self_attn.v_proj.",
                ".attn_out.": ".self_attn.o_proj.",
                "model.transformer.ff_out": "lm_head"
            },
        )

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head"]
                           if self.config.weight_tying else None),
        )

        return loader.load_weights(weights=weights, mapper=mapper)