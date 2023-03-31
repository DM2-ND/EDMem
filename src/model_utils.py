import torch
from copy import deepcopy
from typing import Optional, Union, List, Dict
from transformers import BartConfig


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def shift_tokens_right(input_ids: Union[List, torch.Tensor], pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    if isinstance(input_ids, torch.Tensor):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    elif isinstance(input_ids, list):
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")

        shifted_input_ids = deepcopy(input_ids)
        for i in range(len(shifted_input_ids)):
            shifted_input_ids[i] = [decoder_start_token_id] + shifted_input_ids[i][:-1]
            # replace possible -100 values in labels by `pad_token_id`
            shifted_input_ids[i] = [x if x != -100 else pad_token_id for x in shifted_input_ids[i]]
    else:
        raise TypeError(f"Invalid input type! Expected tensor or list, got {type(input_ids)} instead.")

    return shifted_input_ids


def find_instance_last_entity(input_ids: torch.Tensor, entity_start_token_id):
    """
    Given a batch of input token ids, find how many entities each instance has.
    """
    entity_range = []
    last_entity = []
    entity_iter = 0

    for sequence in input_ids:
        entity_positions = torch.nonzero(sequence == entity_start_token_id)
        num_entities = len(entity_positions)
        entity_range.append((entity_iter, entity_iter + num_entities))
        entity_iter += num_entities
        if num_entities > 0:
            last_entity.append(entity_iter - 1)
        else:
            last_entity.append(None)

    return entity_range, last_entity


def set_autoencoder_config(config: BartConfig):
    """
    For the autoencoder model, we need to set some hyper-parameters using BERT-base's setting
    """
    assert isinstance(config, BartConfig)
    config.d_model = 768
    config.encoder_ffn_dim = 3072
    config.encoder_attention_heads = 12
    return config


def squeeze_beam_inputs(model_inputs: Dict, batch_size: int, num_beams: int):
    idx_each_instance = [i * num_beams for i in range(batch_size)]

    attention_mask = model_inputs["attention_mask"]
    assert attention_mask.size(0) == batch_size * num_beams
    model_inputs["attention_mask"] = attention_mask[idx_each_instance, :]

    encoder_outputs = model_inputs["encoder_outputs"]
    lower_encoder_output = encoder_outputs["lower_encoder_output"]
    encoder_outputs["lower_encoder_output"] = lower_encoder_output[idx_each_instance, :, :]
    upper_encoder_output = encoder_outputs["upper_encoder_output"]
    encoder_outputs["upper_encoder_output"] = upper_encoder_output[idx_each_instance, :, :]
    model_inputs["encoder_outputs"] = encoder_outputs

    return model_inputs

#
# input_ids = [
#     [5, 6, 7, 8, 9, 10],
#     [10, 11, -100, -100, -100]
# ]
# shift_tokens_right(input_ids, 1, 2)
