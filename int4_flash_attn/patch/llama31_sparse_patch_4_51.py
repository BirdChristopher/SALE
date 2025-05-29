import math
import torch
from torch import nn
from typing import Optional, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
import matplotlib.pyplot as plt
import os
import types
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaModel,
    repeat_kv,
    apply_rotary_pos_emb
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from torch.nn.attention import SDPBackend, sdpa_kernel
from functools import partial
from loguru import logger

def layer2device(idx, layer_cnt):
    gpu_in_use = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    step = math.ceil(layer_cnt / gpu_in_use)
    return torch.device(f"cuda:{idx //step}")

def get_device(layer:nn.Module):
    for param in layer.parameters():
        return param.device
    
def make_causal_mask(seq_len, dtype, device):
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full(
        (seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device
    )
    seqlen_arange = torch.arange(seq_len, device=device)
    exclude_mask = seqlen_arange.unsqueeze(0) > seqlen_arange.reshape(-1, 1)
    causal_mask *= exclude_mask
    causal_mask = causal_mask[None, None, :, :]
    return causal_mask

# x.shape [bsz, q_len, hidden_size]
def gather_selected(x, selected_indices):
    return torch.gather(x, 1, selected_indices)

def scatter_add_selected(x, subset, selected_indices):
    return torch.scatter_add(x, 1, selected_indices, subset)

def LlamaAttentionPatch_kv_retrieve_calibration(attn: LlamaAttention, config, idx):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def prefill_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "Do not support bsz > 1 yet."

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        head_group = self.num_heads // self.num_key_value_heads

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # It essentially concat new key/value tensor with the history tensor.
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, None)
        

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                scale = 1 / math.sqrt(query_states.shape[-1]),
                is_causal=True,
                enable_gqa=True
            )
        
        error_threshold = self.error * q_len
        print(f"layer {self.idx} begin")
        
        from int4_flash_attn.flash_attn_int4_skip import flash_attn_func_int4_skip
        # output calibration
        for h_idx in range(0, self.num_heads):
            ref_output = attn_output[:, h_idx:h_idx+1]
            query_part = query_states[:, h_idx:h_idx+1].contiguous()
            key_part = key_states[:, h_idx // head_group : h_idx // head_group + 1].contiguous()
            value_part = value_states[:, h_idx // head_group : h_idx // head_group + 1].contiguous()
            while True:
                result = flash_attn_func_int4_skip(
                    query_part, key_part, value_part, 
                    causal=True,
                    block_q_size=64,
                    block_k_size=32,
                    block_seg_size=128,
                    debug=False,
                    impl_version=2,
                    quant_mode=3,
                    thresholds = torch.tensor([self.dynamic_threshold[h_idx].item()], dtype=torch.float, device=query_states.device)
                )
                error = (result.to(torch.float) - ref_output.to(torch.float)).abs().sum()
                if error < error_threshold or self.dynamic_threshold[h_idx].item() == 0:
                    break
                else:
                    self.dynamic_threshold[h_idx] /= 2         
                    if self.dynamic_threshold[h_idx].item() <= 0.0001:
                        self.dynamic_threshold[h_idx] = 0
    
        assert attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim), attn_output.shape

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
 
    def forward(
        self,
        **kwargs
    ):
        if kwargs["position_ids"].nelement() != 1:
            self.fwd_cnt = 0
            return self.prefill_forward(**kwargs)
        
    attn.forward = types.MethodType(forward, attn)
    attn.prefill_forward = types.MethodType(prefill_forward, attn)
    attn.fwd_cnt = 0
    attn.idx = idx
    attn.threshold = config.threshold
    attn.error = config.error
    attn.num_heads = config.num_attention_heads
    attn.num_key_value_heads = config.num_key_value_heads 
    attn.dynamic_threshold = torch.zeros([attn.num_heads], dtype=torch.float).fill_(0.008)
    attn.hidden_size = config.hidden_size
    attn.lse_record = []
    attn.attn_score_record = []
    attn.coverage_record = []
    

def LlamaAttentionPatch(attn: LlamaAttention, config, idx):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def prefill_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "Do not support bsz > 1 yet."

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # It essentially concat new key/value tensor with the history tensor.
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, None)
        
        attn_output = torch.zeros_like(query_states)
        key_states = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
        value_states = repeat_kv(value_states,self.num_heads // self.num_key_value_heads)

        if os.environ.get("TIMER", "0") == "1":
            self.attn_timer_start.record(stream = torch.cuda.current_stream(key_states.device))
        
        if self.compressor == "original" or q_len <= 1024:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query=query_states,
                    key=key_states,
                    value=value_states,
                    scale = 1 / math.sqrt(query_states.shape[-1]),
                    is_causal=True,
                    enable_gqa=True
                )
        else:
            if self.idx >= self.start_layer:
                # Not Aligned Address.
                attn_output = self.sparse_attn_kernel(
                                query_states, 
                                key_states, 
                                value_states
                            )
            else:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query=query_states,
                        key=key_states,
                        value=value_states,
                        scale = 1 / math.sqrt(query_states.shape[-1]),
                        is_causal=True,
                        enable_gqa=True
                    )
        if os.environ.get("TIMER", "0") == "1":
            self.attn_timer_end.record(stream = torch.cuda.current_stream(key_states.device))

        assert attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim), attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def decoding_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "Do not support bsz > 1 yet."

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # It essentially concat new key/value tensor with the history tensor.
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, None)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weight = query_states @ key_states.transpose(2,3)
        attn_weight = attn_weight / math.sqrt(self.head_dim)
        # attn_weight = attn_weight + attention_mask # NOTE: apply causal mask
        attn_weights = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        assert attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
        
    def forward(
        self,
        **kwargs
    ):
        if kwargs["position_ids"].nelement() != 1:
            self.fwd_cnt = 0
            return self.prefill_forward(**kwargs)
        else:
            self.fwd_cnt += 1
            return self.decoding_forward(**kwargs)
    

    attn.sparsity_threshold = config.sparsity_threshold
    attn.forward = types.MethodType(forward, attn)
    attn.prefill_forward = types.MethodType(prefill_forward, attn)
    attn.decoding_forward = types.MethodType(decoding_forward, attn)
    attn.fwd_cnt = 0
    attn.idx = idx
    attn.start_layer = config.start_layer
    attn.attn_timer_start = torch.cuda.Event(enable_timing=True)
    attn.attn_timer_end = torch.cuda.Event(enable_timing=True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    attn.num_heads = config.num_attention_heads // world_size
    attn.num_key_value_heads = config.num_key_value_heads // world_size
    attn.hidden_size = config.hidden_size // world_size
    print_recall = True if os.environ.get("PRINT_RECALL","0") == "1" else False
    print_quant_error = True if os.environ.get("SHOW_Q_ERROR","0") == "1" else False
    attn.compressor = config.compressor
    # device = layer2device(idx, config.num_hidden_layers)
    if config.compressor == "int4_skip_cuda":
        attn.BLOCK_Q_LEN = config.block_size_q
        attn.BLOCK_K_LEN = config.block_size_k
        threshold_tensor = config.sparsity[idx, :].to(get_device(attn.q_proj))
        attn.sparsity_threshold = 0.0
        sage_up = os.environ.get("SAGE_UP", "0") == "1"
        if sage_up:
            from int4_flash_attn import flash_attn_func_int4_skip_sage as int4_kernel
        else:
            from int4_flash_attn import flash_attn_func_int4_skip as int4_kernel
        attn.sparse_attn_kernel = partial(
            int4_kernel,
            block_q_size = config.block_size_q,
            block_k_size = config.block_size_k,
            thresholds = threshold_tensor,
            local_window = config.local_window,
            block_seg_size = config.block_seg,
            impl_version = config.impl_version,
            quant_mode = config.quant_mode,
            causal = True,
            debug = os.environ.get("DEBUG", "0") == "1"
        )
        logger.info(f"Using int4 skip CUDA kernel to do prefill! Version:{config.impl_version}, two pass? {config.impl_version > 1}, SAGE UP? {sage_up}")
    
    
def LlamaDecoderLayerPatch(layer: LlamaDecoderLayer, config, layer_idx):
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                **kwargs,):

        residual = hidden_states.clone()
        batch, seq_len, embed_dim = hidden_states.shape
        # hidden_states = self.input_layernorm(hidden_states)
        for start_idx in range(0, seq_len, 32000):
            end_idx = min(seq_len, start_idx + 32000)
            hidden_states[:, start_idx:end_idx, :] = self.input_layernorm(
                hidden_states[:, start_idx:end_idx, :]
            )

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states
        
        # Fully Connected
        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)

        # hidden_states = self.mlp(hidden_states)
        
        # hidden_states = residual + hidden_states
        for start_idx in range(0, seq_len, 32000):
            end_idx = min(seq_len, start_idx + 32000)
            part_hidden_states = hidden_states[:, start_idx:end_idx, :].clone()
            part_hidden_states = self.post_attention_layernorm(part_hidden_states)
            part_hidden_states = self.mlp(part_hidden_states)
            hidden_states[:, start_idx:end_idx, :] += part_hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        return outputs, position_ids

    layer.forward = types.MethodType(forward, layer)
    layer.device = get_device(layer.self_attn.q_proj)
    layer.layer_idx = layer_idx

    attn_patch = os.environ.get("CALIBRATION", "original")
    if attn_patch == "original":
        LlamaAttentionPatch(layer.self_attn, config, layer_idx)
    elif attn_patch == "calib_error":
        LlamaAttentionPatch_kv_retrieve_calibration(layer.self_attn, config, layer_idx)
    return layer.half()

def PPLlamaModelPatch(model: LlamaModel, config):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert inputs_embeds is None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = None # NOTE: disable Attention mask.

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = None
            if past_key_values is not None:
                past_key_value = past_key_values[idx] if len(past_key_values) == len(self.layers) else None
                # assert past_key_values[idx][0].device == get_device(decoder_layer)
            
            if os.environ.get("WORLD_SIZE", "1") == "1" and hidden_states.device != decoder_layer.device:
                hidden_states = hidden_states.to(decoder_layer.device)

            if os.environ.get("WORLD_SIZE", "1") == "1" and position_ids.device != decoder_layer.device:
                position_ids = position_ids.to(decoder_layer.device)

            if os.environ.get("WORLD_SIZE", "1") == "1" and position_embeddings[0].device != decoder_layer.device:   
                position_embeddings = (
                    position_embeddings[0].to(decoder_layer.device),
                    position_embeddings[1].to(decoder_layer.device)
                )
            
            torch.cuda.set_device(hidden_states.device.index)

            layer_outputs, position_ids = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if os.environ.get("WORLD_SIZE", "1") == "1" and hidden_states.device != get_device(self.norm):
            hidden_states = hidden_states.to(get_device(self.norm))
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    model.vocab_size = config.vocab_size
    model.forward = types.MethodType(forward, model) 
    if os.environ.get("WORLD_SIZE", "1") == "1": # Scattered model scenario
        model.embed_tokens = model.embed_tokens.to(torch.device("cuda:0"))
        for i in range(config.num_hidden_layers):
            model.layers[i] = LlamaDecoderLayerPatch(model.layers[i].to(layer2device(i, config.num_hidden_layers)), config, i)
        model.norm = model.norm.to(torch.device(f"cuda:{config.pp_size-1}"))
    else:
        for i in range(config.num_hidden_layers):
            model.layers[i] = LlamaDecoderLayerPatch(model.layers[i], config, i)
    
    model.gradient_checkpointing = False
    # Initialize weights and apply final processing
    return model.half()

class KwargsForCausalLM(FlashAttentionKwargs): ...

def LlamaForCausalLMPatch(M: LlamaForCausalLM, config):
    def analyse_sparsity(
        self, coverage_threshold
    ):
        threshes = []
        for layer in self.model.layers:
            threshes.append(layer.self_attn.dynamic_threshold[None,:])
        result = torch.concat(threshes, dim = 0)
        print(result)
        print(torch.nonzero(result,as_tuple=True)[0].shape[0], "heads are sparse")
        return result, None

    # Copied from transformers.models.Llama.modeling_Llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        ##If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        is_decoding = (cache_position.nelement() == 1)
        
        if past_key_values is not None:
            assert inputs_embeds is None
            if is_decoding:
                assert input_ids.shape[1] != cache_position.shape[0]  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
            else:
                assert input_ids.shape[1] == cache_position.shape[0]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            assert len(attention_mask.shape) == 2 and attention_mask.shape[0] == 1, attention_mask.shape
            assert attention_mask.nelement() == (cache_position[-1].item()+1), f"{attention_mask.nelement()},{cache_position}"
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    
        if input_ids.shape[-1] > 1:
            # attention_mask = make_causal_mask(input_ids.shape[-1], torch.bfloat16, "cuda:0")
            attention_mask = None
        else:
            attention_mask = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs
        )

        hidden_states = outputs[0]
        # logits = self.lm_head(hidden_states)
        # logits = logits.float()
        logits = self.lm_head(hidden_states[:,-1:,:])
        logits = logits.float()
        
        if os.environ.get("WORLD_SIZE", "1") == "1": # Scattered model scenario
            if outputs.hidden_states is not None:
                outputs.hidden_states = outputs.hidden_state.to(torch.device(f"cuda:0"))
            logits = logits.to(torch.device(f"cuda:0"))
            if outputs.attentions is not None:
                outputs.attentions = outputs.attentions.to(torch.device(f"cuda:0"))

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    M.model = PPLlamaModelPatch(M.model, config)
    M.forward = types.MethodType(forward, M) 
    M.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, M) 
    M.analyse_sparsity = types.MethodType(analyse_sparsity, M)
    if os.environ.get("WORLD_SIZE", "1") == "1": # Scattered model scenario
        M.lm_head = M.lm_head.to(torch.device(f"cuda:{config.pp_size - 1}")).half()
        M.model.embed_tokens = M.model.embed_tokens.to(torch.device(f"cuda:0")).half()
    M.layer_num = config.num_hidden_layers
    M.kv_head_cnt = config.num_key_value_heads
    return M
    # self.post_init()
