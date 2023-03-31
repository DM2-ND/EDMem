"""
Other modules used in the model.
@Date  : 01/24/2022
@Author: Zhihan Zhang
@mail  : zzhang23@nd.edu
@homepage: ytyz1307zzh.github.io
"""

import math
import pdb
from typing import List, Callable
import torch
import torch.nn as nn
from copy import deepcopy
from transformers.generation_logits_process import LogitsProcessor
from transformers.activations import ACT2FN


class Linear(nn.Module):
    """
    Simple Linear layer with xavier init
    """
    def __init__(self, d_in: int, d_out: int, dropout: float, bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.linear(x))


class EMAGPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0, attention_mask=None):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        # positions = torch.arange(
        #     past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        # )
        if attention_mask is None:
            positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device)
        else:
            positions = (attention_mask.long().cumsum(-1) - 1)
            positions.masked_fill_(attention_mask == 0, 0)  # can be filled with anything >= 0

        positions = positions + past_key_values_length

        return super().forward(positions + self.offset)


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned
    constrained generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more
    information.

    Args:
        prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed
            tokens for the next generation step conditioned on the previously generated tokens `inputs_ids` and
            the batch ID `batch_id`.
    """

    def __init__(
            self,
            args,
            data_ids: List[str],
            prefix_allowed_tokens_fn: Callable[[List[str], int, torch.Tensor], List[int]],
            num_beams: int,
            use_instance_id: bool = False
    ):
        self.data_ids = data_ids
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.rescale_logits = args.rescale_logits
        self.use_instance_id = use_instance_id

    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        scores: for beam search, it is the log-softmax score, range (-inf, 0];
                for greedy search, it is the raw logits generated by the model
        logits: the raw logits generated by the model
        """
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.use_instance_id:
                    instance_id = batch_id * self._num_beams + beam_id
                    allowed_tokens = self._prefix_allowed_tokens_fn(self.data_ids, instance_id, sent)
                else:
                    allowed_tokens = self._prefix_allowed_tokens_fn(self.data_ids, batch_id, sent)
                mask[batch_id * self._num_beams + beam_id, allowed_tokens] = 0
                if self._num_beams > 1 and len(allowed_tokens) < mask.size(-1) - 1:
                    if self.rescale_logits == "norm":
                        scores = self.adjust_beam_score(
                            idx=batch_id * self._num_beams + beam_id,
                            scores=scores,
                            allowed_tokens=allowed_tokens
                        )
        if self._num_beams > 1 and self.rescale_logits == "softmax":
            logits = logits + mask
            scores = nn.functional.log_softmax(logits, dim=-1)

        return scores + mask

    def adjust_beam_score(self, idx, scores, allowed_tokens):
        """
        Adjust the beam scores of allowed_tokens, since many tokens in the original distribution are masked
        scores: original beam scores, (batch * beam, vocab_size)
        """
        num_allowed_tokens = len(allowed_tokens)
        allowed_token_scores = deepcopy(scores[idx, allowed_tokens])
        original_topk_scores = scores[idx].topk(k=num_allowed_tokens).values

        assert self.rescale_logits == "norm"

        allowed_token_scores = self.rescale_norm(allowed_token_scores, original_topk_scores)
        # elif self.rescale_logits == "softmax":
        #     allowed_token_scores = nn.functional.log_softmax(torch.exp(allowed_token_scores))

        scores[idx, allowed_tokens] = allowed_token_scores
        return scores

    def rescale_norm(self, dist_a: torch.Tensor, dist_b: torch.Tensor):
        """
        Re-scale the distribution A to the scale of distribution B
        """
        assert len(dist_a) == len(dist_b)
        # If only one item in the distribution, directly copy the value
        if len(dist_a) == 1:
            dist_a = dist_b
            return dist_a

        max_a = max(dist_a)
        max_b = max(dist_b)

        min_a = min(dist_a)
        min_b = min(dist_b)

        diff_a = max_a - min_a
        diff_b = max_b - min_b

        dist_a = (dist_a - min_a) * diff_b / diff_a + min_b
        return dist_a


class LMHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.d_model, config.d_model, dropout=config.dropout)
        if isinstance(config.activation_function, str):
            self.transform_act_fn = ACT2FN[config.activation_function]
        else:
            self.transform_act_fn = config.activation_function
        self.LayerNorm = nn.LayerNorm(config.d_model)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = LMHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
