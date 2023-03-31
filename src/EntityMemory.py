"""
The entity memory module.
@Date  : 01/24/2022
@Author: Zhihan Zhang
@mail  : zzhang23@nd.edu
@homepage: ytyz1307zzh.github.io
"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules import Linear
from typing import Tuple, Optional
from Constants import *


class EntityMemory(nn.Module):
    """
    Entity Memory module.
    """

    def __init__(
        self,
        config,
        model_args,
        entity_embedding: nn.Linear,
        output_embedding: bool = True
    ):
        """
        :param entity_embedding: entity embedding module, should be a nn.Linear initialized outside
        :param output_embedding: if True, then output the weighted knowledge embedding
        """
        super().__init__()
        self.config = config
        self.args = model_args
        self.fp16 = model_args.fp16
        self.d_model = config.d_model
        self.entity_vocab_size = model_args.entity_vocab_size
        self.d_entity_embed = model_args.entity_embed_size
        self.dropout_rate = config.dropout
        self.output_embedding = output_embedding

        self.InputLayer = Linear(d_in=self.d_model, d_out=self.d_entity_embed, dropout=self.dropout_rate)
        self.EntityEmbedding = entity_embedding

        # initialize input linear layer
        self._init_weights(self.InputLayer)

        if output_embedding:
            self.OutputLayer = Linear(d_in=self.d_entity_embed, d_out=self.d_model, dropout=self.dropout_rate)
            self._init_weights(self.OutputLayer)

        self.ELLoss = nn.NLLLoss(ignore_index=NIL_ENTITY)

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_state: torch.Tensor,
        entity_link: torch.Tensor = None,
        entity_mask: torch.Tensor = None,
        topk: int = 100,
        last_entity_only: bool = False
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        :param input_ids: input token ids to the model, shape (batch, seq)
        :param hidden_state: hidden state of last layer in the lower encoder, shape (batch, seq, d_model)
        :param entity_link: input vector indicating the linked entity in each position.
                            Non-[E_s] tokens are labeled as NIL_ENTITY (see Constants.py),
                            shape (batch, seq), elements are entity ids or NIL_ENTITY.
                            Will be None during inference.
        :param entity_mask: a 0-1 vector indicating which entities to mask. Masked entities will receive an attention
                            score of -inf (which lead to an attention weight of 0)
        NOTE: the second dimension (sequence length) of hidden_state and entity_link can be different in decoder.
        :param topk: how many entities to retrieve before doing attention (during inference)
        :param last_entity_only: whether to only access memory on the last mention (for entity linking task)
        :return: (loss, attention output, topk_indices), loss is the NLLLoss of attention probability of
                 the ground-truth entity, and attention output is the integrated knowledge embedding
                 (output of the entity memory module), with shape (batch, seq, d_model),
                 topk_indices is the indices of top-k nearest entities
        """

        # prepare the output tensor
        memory_output = torch.zeros_like(hidden_state).to(hidden_state.device)
        # if fp16 is used, convert this tensor to float16 type
        if self.fp16:
            memory_output = memory_output.to(torch.float16)

        # If past_entity_only = True, it means we are performing entity linking task instead of generation task.
        batch_size, seq_length = input_ids.size()
        if last_entity_only:
            start_position = torch.LongTensor([[i, seq_length - 1] for i in range(batch_size)]).to(hidden_state.device)
        # get the start positions of entity mentions, shape (num_mentions, 2)
        else:
            start_position = torch.nonzero(input_ids == self.args.entity_start_token_id)
        assert len(start_position.size()) == 2
        assert start_position.size(-1) == len(input_ids.size())
        num_mentions = start_position.size(0)

        # if there is no entity mention in input text,
        # return zero for the loss value and all-zero matrix for memory_output
        # if num_mentions == 0:
        #     loss = torch.tensor(0.0).to(hidden_state.device)
        #     if self.fp16:
        #         loss = loss.to(torch.float16)
        #     return loss, memory_output, None

        # transpose start_position to get indices for respective dimensions, shape (2, num_mentions)
        start_position = start_position.transpose(0, 1)
        # double check
        entity_start_ids = input_ids[start_position[0], start_position[1]]
        assert torch.all(entity_start_ids == self.args.entity_start_token_id)

        # get the embeddings of indexed positions, i.e., embeddings of <E_s> positions, shape (num_mentions, d_model)
        mention_embedding = hidden_state[start_position[0], start_position[1]]

        # convert d_model dimension embedding to d_entity_embed dimension, shape (num_mentions, d_entity_embed)
        memory_query = self.InputLayer(mention_embedding)
        assert memory_query.size() == (num_mentions, self.d_entity_embed)

        # Get attention scores by calculating dot product between the query and entity embeddings
        attention_score = memory_query.matmul(self.EntityEmbedding.weight)
        assert attention_score.size() == (num_mentions, self.entity_vocab_size)

        if entity_mask is not None:
            assert entity_mask.size() == (self.entity_vocab_size,)

            # convert masked positions from 1.0 to -inf, to prepare for updating attention scores to -inf
            entity_mask = entity_mask.float()
            # entity_mask.masked_fill_(entity_mask.bool(), torch.finfo(entity_mask.dtype).min)
            attention_score.masked_fill_(entity_mask.bool(), torch.finfo(entity_mask.dtype).min)

        topk_entity_indices = None
        # during training, use the whole 1M embeddings to compute attention
        if self.training:
            # NOTE: EntityEmbedding.weight is (d_entity_embed, num_entities)
            # attention weights, shape (num_mentions, num_entities)
            attention_weight = F.softmax(attention_score, dim=1)
            assert attention_weight.size() == (num_mentions, self.entity_vocab_size)

            # get the ground-truth entity ids for each annotated mention (stored in the same position as <E_s>)
            gold_entity = entity_link[start_position[0], start_position[1]]
            assert gold_entity.size() == (num_mentions,)

            # entity linking loss
            if entity_mask is not None:
                # If some entity are masked, do not apply loss on those zero attention weights
                select_attention_weight = attention_weight[:, ~entity_mask.bool()]
                loss = self.ELLoss(torch.log(select_attention_weight), gold_entity)
            else:
                loss = self.ELLoss(torch.log(attention_weight), gold_entity)

            if num_mentions == 0:
                loss = torch.nan_to_num(loss, nan=0.0)

            pick_entity_embedding = self.EntityEmbedding(attention_weight)

        # during inference, use top-k nearest embeddings to compute attention
        else:
            loss = None
            pick_entity_embedding, topk_entity_indices = self.get_k_nearest(attention_score, topk)

        assert pick_entity_embedding.size() == (num_mentions, self.d_entity_embed)

        if self.output_embedding:
            # do dimension conversion from d_entity_embed to d_model,
            # and fill in the corresponding positions (<E_s> positions) of the prepared memory output matrix.
            memory_output[start_position[0], start_position[1]] = self.OutputLayer(pick_entity_embedding)
            return loss, memory_output, topk_entity_indices
        else:
            return loss, None, topk_entity_indices

    def get_k_nearest(self, attention_score: torch.Tensor, k: int) -> (torch.Tensor, torch.Tensor):
        """
        Only use top-k entity entries to calculate attention
        :param attention_score: attention scores, size (num_mentions, entity_vocab_size)
        :param k: the number of entity entries to qualify into attention calculation
        :return pick_entity_embedding: the attention weighted embedding of top-k entities
        :return topk_indices: the indices of top-k entities
        """
        num_mentions = attention_score.size(0)
        assert attention_score.size(1) == self.entity_vocab_size

        # top-k similarity scores and indices, shape (num_mentions, k)
        topk_values, topk_indices = torch.topk(attention_score, k, dim=1)
        attention_weight = F.softmax(topk_values, dim=1)

        # matrix 1: (num_mentions, d_entity_embed, k), the transpose of top-k embeddings
        # matrix 2: (num_mentions, k, 1), the attention weights
        # result: (num_mentions, d_entity_embed, 1), the weighted sum of top-k embeddings
        pick_entity_embedding = torch.bmm(
            self.EntityEmbedding.weight[:, topk_indices].transpose(0, 1),
            attention_weight.view(num_mentions, k, 1)
        )

        # squeeze to get the final pick_entity_embedding, shape (num_mentions, d_entity_embed)
        pick_entity_embedding.squeeze_(dim=-1)
        return pick_entity_embedding, topk_indices
