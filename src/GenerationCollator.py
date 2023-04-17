import pdb
from typing import Union, Any, Tuple, Optional, List, Dict
from dataclasses import dataclass
from copy import deepcopy
import torch
from transformers import PretrainedConfig
from model_utils import shift_tokens_right
from Constants import NUM_ADDED_VOCAB, NIL_ENTITY, ENTITY_START_TOKEN, ENTITY_END_TOKEN


@dataclass
class GenerationCollator:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling.
        ssm (`bool`, *optional*, defaults to `True`):
            Whether or not to use salient span masking.
        mlm_probability (`float`, *optional*, defaults to 0.1):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        ssm_probability (`float`, *optional*, defaults to 0.2):
            The probability with which to mask salient spans in the input, when `ssm` is set to `True`.
        loss_on_mask_tokens (`bool`, *optional*, defaults to False):
            If `True`, then only compute loss on masked tokens
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a
    [`PreTrainedTokenizer`] or a [`PreTrainedTokenizerFast`] with the
    argument `return_special_tokens_mask=True`.

    </Tip>"""

    config: PretrainedConfig
    task_name: str
    generate_target: str = "target"
    max_seq_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        pass

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        :param examples: A list of items in the dataset. Each item is a dictionary of input fields
        """
        pad_token_map = {
            "input_ids": self.config.pad_token_id,
            "decoder_input_ids": self.config.pad_token_id,
            # "attention_mask": 0,
            "labels": self.config.pad_token_id,
            "input_entity_link": NIL_ENTITY,
            "output_entity_link": NIL_ENTITY,
        }
        pad_direction_map = {
            "input_ids": "right",
            "decoder_input_ids": "left",
            "input_entity_link": "right",
        }

        batch_size = len(examples)

        batch = {key: [exp[key] for exp in examples] for key in examples[0].keys()}

        # If there is a prefix at decoder side, we need to shift the tokens to the right
        #  to align "decoder_input_ids" and "labels"
        if self.generate_target == "prefix":
            assert "decoder_input_ids" in batch.keys()
            batch["decoder_input_ids"] = shift_tokens_right(
                input_ids=batch["decoder_input_ids"],
                pad_token_id=self.config.pad_token_id,
                decoder_start_token_id=self.config.decoder_start_token_id
            )
            # In training, we also need to shift output_entity_link because we won't do it inside the model
            #  when decoder_input_ids is not None (see Model.py)
            if "output_entity_link" in batch.keys():
                batch["output_entity_link"] = shift_tokens_right(
                    input_ids=batch["output_entity_link"],
                    pad_token_id=NIL_ENTITY,
                    decoder_start_token_id=NIL_ENTITY
                )

            # If "decoder_input_ids" is part of input, then "output_entity_link" and "labels" should pad at the left
            if self.task_name == "creak":
                # In creak, the prefices are of same length, so we follow the traditional setting to pad on right
                pad_direction_map["decoder_input_ids"] = "right"
                pad_direction_map["output_entity_link"] = "right"
                pad_direction_map["labels"] = "right"
            else:
                raise NotImplementedError(f"Task {self.task_name} does not need a prefix!")

        elif self.generate_target == "target":
            pad_direction_map["output_entity_link"] = "right"
            pad_direction_map["labels"] = "right"
        elif self.generate_target == "entity_linking":
            pad_direction_map["output_entity_link"] = "right"
            pad_direction_map["labels"] = "right"
        else:
            raise ValueError("Invalid generate_target value!")

        for field_name, field_value in batch.items():
            # some data fields is not in pad_token_map, which means they don't need padding (e.g., metadata)
            if field_name in pad_token_map.keys():
                pad_token = pad_token_map[field_name]
                pad_direction = pad_direction_map[field_name]
                padded_value = self.pad_batch_input(field_value,
                                                    pad_token=pad_token,
                                                    pad_direction=pad_direction,
                                                    max_seq_length=self.max_seq_length,
                                                    pad_to_multiple_of=self.pad_to_multiple_of)
                batch[field_name] = padded_value

        # Get the other input fields on-the-fly
        batch["attention_mask"] = (batch["input_ids"] != self.config.pad_token_id).to(torch.long)
        if "decoder_input_ids" in batch.keys():
            batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.config.pad_token_id).to(torch.long)
        # Ignore the pad tokens in loss computing by setting them to -100 (the ignore idex of CrossEntropyLoss)
        if "labels" in batch:
            batch["labels"][batch["labels"] == self.config.pad_token_id] = -100

        return batch

    def pad_batch_input(self,
                        batch_input: List[List[int]],
                        pad_token: int,
                        pad_direction: str = "right",
                        max_seq_length: Union[int, None] = None,
                        pad_to_multiple_of: Union[int, None] = None):
        """
        Pad a batch of input data and return a tensor.
        Do dynamic padding: sequences are pad to the longest sequence length in the batch.
        :param batch_input: the batched input
        :param pad_token: the token to act as padding token
        :param max_seq_length: the specified length to pad to.
        :param pad_to_multiple_of: whether to pad the data to multiples of a given integer length.
        """
        if max_seq_length is None:
            max_length = max([len(l) for l in batch_input])
        else:
            max_length = max_seq_length

        # if pad_to_multiple_of is specified, max_length is set to the smallest multiple of the given
        # interger length which is larger than max_length (if a static max_seq_length is set, ignore this)
        if pad_to_multiple_of is not None and \
                max_seq_length is None and \
                (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        padded_batch = []
        for instance in batch_input:
            difference = max_length - len(instance)
            if pad_direction == "right":
                padded_instance = instance + [pad_token] * difference
            elif pad_direction == "left":
                padded_instance = [pad_token] * difference + instance
            else:
                raise ValueError(f"Invalid padding direction! Should be `left` or `right`, but got {pad_direction}.")
            padded_batch.append(padded_instance)

        padded_tensor = torch.LongTensor(padded_batch)
        assert padded_tensor.shape == (len(batch_input), max_length)
        return padded_tensor
