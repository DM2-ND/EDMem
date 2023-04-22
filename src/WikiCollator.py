import pdb
from typing import Union, Any, Tuple, Optional, List, Dict
from dataclasses import dataclass
from copy import deepcopy
import torch
from transformers import PreTrainedTokenizerBase
from Constants import NUM_ADDED_VOCAB, NIL_ENTITY, ENTITY_START_TOKEN, ENTITY_END_TOKEN


@dataclass
class WikiCollator:
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

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    ssm: bool = True
    mlm_probability: float = 0.1
    ssm_probability: float = 0.2
    mlm_random_replace: float = 0.0
    loss_on_mask_tokens: bool = False
    max_seq_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    no_memory: Optional[bool] = False

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        :param examples: A list of items in the dataset. Each item is a dictionary of input fields
        """
        pad_token_map = {
            "input_ids": self.tokenizer.pad_token_id,
            # "attention_mask": 0,
            # "labels": self.tokenizer.pad_token_id,
            "input_entity_link": NIL_ENTITY,
            # "output_entity_link": NIL_ENTITY,
        }

        batch = {key: [exp[key] for exp in examples] for key in examples[0].keys()}

        for field_name, field_value in batch.items():
            # some data fields is not in pad_token_map, which means they don't need padding (e.g., metadata)
            if field_name in pad_token_map.keys():
                pad_token = pad_token_map[field_name]
                padded_value = self.pad_batch_input(field_value,
                                                    pad_token=pad_token,
                                                    max_seq_length=self.max_seq_length,
                                                    pad_to_multiple_of=self.pad_to_multiple_of)
                batch[field_name] = padded_value

        # Get the other input fields on-the-fly
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).to(torch.long)
        batch["labels"] = deepcopy(batch["input_ids"])
        batch["output_entity_link"] = deepcopy(batch["input_entity_link"])

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        original_inputs = deepcopy(batch["input_ids"])
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mlm_masking(
                batch["input_ids"], batch["labels"], special_tokens_mask=special_tokens_mask
            )

        if self.ssm:
            batch["input_ids"], batch["labels"], mention_mask_result = self.ssm_masking(
                batch["input_ids"], original_inputs, batch["labels"]
            )
            batch["entity_mention_mask"] = mention_mask_result

        if not self.mlm and not self.ssm:
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100

        if self.no_memory:
            batch["input_ids"], batch["labels"] = self.remove_entity_tokens(batch["input_ids"], batch["labels"])
            batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).to(torch.long)
            assert batch["input_ids"].shape == batch["labels"].shape
            batch_keys = list(batch.keys())
            for key in batch_keys:
                if key not in ["input_ids", "attention_mask", "labels"]:
                    batch.pop(key)

        return batch

    def pad_batch_input(self,
                        batch_input: List[List[int]],
                        pad_token: int,
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
            padded_instance = instance + [pad_token] * difference
            padded_batch.append(padded_instance)

        padded_tensor = torch.LongTensor(padded_batch)
        assert padded_tensor.shape == (len(batch_input), max_length)
        return padded_tensor

    def ssm_masking(
            self,
            inputs: torch.Tensor,
            original_inputs: torch.Tensor,
            labels: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for salient span masking
        """
        # Entity-related special token ids
        entity_start_token_id = self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN)
        entity_end_token_id = self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        eos_token_id = self.tokenizer.eos_token_id

        # The unit of masking is the entity mentions
        entity_starts = torch.nonzero(inputs == entity_start_token_id)
        probability_matrix = torch.full((len(entity_starts),), self.ssm_probability)

        mention_mask_result = torch.bernoulli(probability_matrix).bool()  # whether each mention is chosen to be masked
        # Find the indices of those mentions which are chosen to be masked
        masked_mentions = torch.nonzero(mention_mask_result).squeeze(dim=1)
        masked_mention_indices = entity_starts[masked_mentions]

        # Mask out the entire mention following <E_s> until <E_e>
        for mention in masked_mention_indices:
            idx_dim0, idx_dim1 = mention
            idx_dim1 += 1
            while inputs[idx_dim0][idx_dim1] != entity_end_token_id and \
                    inputs[idx_dim0][idx_dim1] != eos_token_id:
                inputs[idx_dim0][idx_dim1] = self.tokenizer.mask_token_id
                idx_dim1 += 1

        # If only compute loss on mask tokens, set the other tokens to -100 to ignore cross-entropy loss
        changed_positions = (inputs != original_inputs)
        if self.loss_on_mask_tokens:
            # If not already did mlm, then the labels here are identical to the inputs
            if not self.mlm:
                labels[inputs != self.tokenizer.mask_token_id] = -100
            # Otherwise, the labels have already been modified in mlm
            else:
                labels[changed_positions] = original_inputs[changed_positions]
        # Otherwise, only ignore padding tokens
        else:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return inputs, labels, mention_mask_result

    def mlm_masking(
            self,
            inputs: torch.Tensor,
            labels: torch.Tensor,
            special_tokens_mask: Optional[Any] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # Special tokens are never masked
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # If only compute loss on mask tokens, set the other tokens to -100 to ignore cross-entropy loss
        if self.loss_on_mask_tokens:
            labels[~masked_indices] = -100
        # Otherwise, only ignore padding tokens
        else:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # (1 - mlm_random_replace) of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        mask_ratio = 1 - self.mlm_random_replace
        indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_ratio)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # The rest of the time, we replace masked input tokens with random word
        indices_random = masked_indices & ~indices_replaced
        # Make sure that <E_s> and <E_e> are not generated in this step
        random_words = torch.randint(len(self.tokenizer) - NUM_ADDED_VOCAB, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def remove_entity_tokens(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        For the no-memory ablation, remove the <E_s> and <E_e> tokens from input and output
        """
        entity_start_token_id = self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN)
        entity_end_token_id = self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        pad_token_id = self.tokenizer.pad_token_id
        batch_size, seq_len = inputs.shape

        inputs = inputs.tolist()
        labels = labels.tolist()

        for i in range(len(inputs)):
            inputs[i] = [x for x in inputs[i] if x != entity_start_token_id and x != entity_end_token_id]
            labels[i] = [x for x in labels[i] if x != entity_start_token_id and x != entity_end_token_id]
            assert len(inputs[i]) == len(labels[i])
            inputs[i] = inputs[i] + [pad_token_id] * (seq_len - len(inputs[i]))
            labels[i] = labels[i] + [-100] * (seq_len - len(labels[i]))

        return torch.LongTensor(inputs), torch.LongTensor(labels)
