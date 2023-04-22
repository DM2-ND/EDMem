"""
Entity memory augmented encoder-decoder transformer model
@Date  : 01/22/2022
@Author: Zhihan Zhang
@mail  : zzhang23@nd.edu
@homepage: ytyz1307zzh.github.io
"""
import pdb
import torch
import torch.nn as nn
import time
import warnings
import torch.nn.functional as F
import torch.distributed as dist
from copy import deepcopy
from typing import Optional, Dict, Tuple, Any, Union, List, Callable, Iterable
from transformers import BertModel, PreTrainedModel, BertConfig
from transformers.utils import logging
from Encoder import EMAGLowerEncoder, EMAGUpperEncoder
from EntityMemory import EntityMemory
from Output import EMAGModelOutput
from Loss import DetailLoss
from Constants import NIL_ENTITY
from Modules import LMHead
from model_utils import shift_tokens_right, find_instance_last_entity

logger = logging.get_logger(__name__)


class EMAGAutoEncoder(PreTrainedModel):
    """
    The main architecture of the EMAG model.
    """
    def __init__(
            self,
            config,
            model_args,
    ):
        super().__init__(config)
        self.config = deepcopy(config)
        self.model_args = deepcopy(model_args)
        self.num_lower_layers = model_args.num_lower_layers
        self.num_upper_layers = model_args.num_upper_layers
        self.entity_vocab_size = model_args.entity_vocab_size
        self.d_model = config.d_model
        self.elloss_weight = model_args.elloss_weight
        self.inference_nearest_k_entity = model_args.inference_nearest_k_entity

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # Entity embedding table. Use nn.Linear to define to allow weighted sum using attention weights.
        self.EntityEmbedding = nn.Linear(
            in_features=self.entity_vocab_size,
            out_features=model_args.entity_embed_size,
            bias=False
        )

        # entity memory module
        self.EncoderEntityMemory = EntityMemory(config, model_args, self.EntityEmbedding)
        self.EntityLinkHead = EntityMemory(config, model_args, self.EntityEmbedding, output_embedding=False)

        # encoder and decoder blocks
        self.LowerEncoder = EMAGLowerEncoder(config, model_args, embed_tokens=self.shared)
        self.UpperEncoder = EMAGUpperEncoder(config, model_args)

        # output heads
        self.LMHead = LMHead(config)

        # tie weights of LMHead and model.shared
        # self.tie_weights()

        # layer norm modules after entity memory access and before upper encoder/decoder
        self.EncoderMemoryLayerNorm = nn.LayerNorm(config.d_model)

        # loss for LM objective
        if model_args.entity_token_weight is not None:
            lm_token_weight = torch.full((vocab_size + 2,), 1.0)
            lm_token_weight[model_args.entity_start_token_id] = model_args.entity_token_weight
            lm_token_weight[model_args.entity_end_token_id] = model_args.entity_token_weight
            self.LMLoss = nn.CrossEntropyLoss(weight=lm_token_weight)
        else:
            self.LMLoss = nn.CrossEntropyLoss()

        # Task-specific arguments for the entity linking task
        self.do_entity_linking = (model_args.generate_target == "entity_linking")
        if self.do_entity_linking:
            self.final_layer_loss = model_args.final_layer_loss
            self.apply_el_loss = model_args.apply_el_loss

        self.pad_token_id = config.pad_token_id

        # Initialize weights and apply final processing
        self.post_init()

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

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.LowerEncoder.embed_tokens = self.shared

    def get_encoder(self):
        return NotImplementedError("get_encoder is ambiguous.")

    def get_decoder(self):
        return NotImplementedError("get_decoder is ambiguous.")

    def get_output_embeddings(self):
        return self.LMHead.decoder

    def set_output_embeddings(self, new_embeddings):
        self.LMHead.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            input_entity_link=None,
            output_entity_link=None,
            entity_mention_mask=None,
            entity_mask=None,
            metadata=None,
            head_mask=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
    ):
        """
        Main parameters:
        :param input_ids: input token ids (pretrain: masked sequence, QA: question), (batch, in_seq_len)
        :param attention_mask: attention mask, (batch, in_seq_len)
        :param labels: target output sequence (pretrain: original sequence, QA: answer, (batch, out_seq_len)
        :param input_entity_link: entity linking labels of the input sequence, (batch, in_seq_len)
        :param output_entity_link: we keep this entry only to align with encoder-decoder pre-training
        :param entity_mention_mask: mask of entity mentions, (num_mentions, ) (ONLY USED IN compute_metrics)
        :param entity_mask: a 0-1 vector indicating which entities to mask. Masked entities will receive an attention
                            score of -inf (which lead to an attention weight of 0)
        :param metadata: meta data, like data id (NOT USED)
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # input and output tensor dimensions
        assert input_ids.shape == attention_mask.shape
        batch_size = input_ids.size(0)
        input_seq_len = input_ids.size(1)

        if labels is not None:
            assert input_ids.shape == input_entity_link.shape

        # top-k nearest entities at memory access
        encoder_topk_entity, final_topk_entity, topk_entity = None, None, None
        # entity linking losses at memory access
        encoder_memory_loss, final_memory_loss = None, None

        # lower encoder
        lower_encoder_output_class = self.LowerEncoder(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       output_attentions=output_attentions,
                                                       output_hidden_states=output_hidden_states,
                                                       return_dict=True)
        lower_encoder_output = lower_encoder_output_class.last_hidden_state
        assert lower_encoder_output.shape == (batch_size, input_seq_len, self.d_model)

        # encoder needs to access entity memory
        if self.training:
            encoder_memory_loss, encoder_entity_embedding, _ = \
                self.EncoderEntityMemory(input_ids=input_ids,
                                         hidden_state=lower_encoder_output,
                                         entity_link=input_entity_link,
                                         entity_mask=entity_mask)
        else:
            _, encoder_entity_embedding, encoder_topk_entity = \
                self.EncoderEntityMemory(input_ids=input_ids,
                                         hidden_state=lower_encoder_output,
                                         entity_link=input_entity_link,
                                         entity_mask=entity_mask,
                                         topk=self.inference_nearest_k_entity)

        # add entity embeddings to token embeddings with a layer norm
        upper_encoder_input = self.EncoderMemoryLayerNorm(encoder_entity_embedding + lower_encoder_output)
        assert upper_encoder_input.shape == (batch_size, input_seq_len, self.d_model)

        # upper encoder
        upper_encoder_output_class = self.UpperEncoder(inputs_embeds=upper_encoder_input,
                                                       attention_mask=attention_mask,
                                                       output_attentions=output_attentions,
                                                       output_hidden_states=output_hidden_states,
                                                       return_dict=True)
        upper_encoder_output = upper_encoder_output_class.last_hidden_state
        assert upper_encoder_output.shape == (batch_size, input_seq_len, self.d_model)

        # language modeling head
        lm_logits = self.LMHead(upper_encoder_output)
        assert lm_logits.shape == (batch_size, input_seq_len, self.shared.num_embeddings)

        # compute the losses: language modeling loss and entity linking loss
        final_loss, lm_loss, el_loss = None, None, None

        if self.do_entity_linking and self.final_layer_loss == "answer_only":
            last_entity_only = True
        elif self.do_entity_linking and not self.training:
            if self.final_layer_loss == "all_eval":
                last_entity_only = False
            elif self.final_layer_loss == "all":
                last_entity_only = True
            else:
                raise ValueError(f"Invalid final_layer_loss value {self.final_layer_loss}")
        else:
            last_entity_only = False

        if self.training:
            final_memory_loss, _, _ = \
                self.EntityLinkHead(input_ids=input_ids,
                                    hidden_state=upper_encoder_output,
                                    entity_link=input_entity_link,
                                    entity_mask=entity_mask,
                                    last_entity_only=last_entity_only)

            # if we only calculate entity linking loss in the last layer, set the encoder loss to zero
            if self.do_entity_linking and self.apply_el_loss == "final":
                el_loss = final_memory_loss
            # otherwise, the entity linking loss is the average of encoder and final-layer losses
            else:
                el_loss = (encoder_memory_loss + final_memory_loss) / 2

            if self.do_entity_linking:
                lm_loss = self.LMLoss(input=lm_logits.view(batch_size * input_seq_len, self.shared.num_embeddings),
                                      target=input_ids.view(batch_size * input_seq_len))
                final_loss = lm_loss * 0 + el_loss
            else:
                lm_loss = self.LMLoss(input=lm_logits.view(batch_size * input_seq_len, self.shared.num_embeddings),
                                      target=labels.view(batch_size * input_seq_len))
                # the final loss value is the weighted sum of lm_loss and el_loss
                final_loss = lm_loss + self.elloss_weight * el_loss

        else:
            # In evaluation mode, labels are available during pre-training, but are not available in fine-tuning
            if labels is not None and not self.do_entity_linking:
                lm_loss = self.LMLoss(input=lm_logits.view(batch_size * input_seq_len, self.shared.num_embeddings),
                                      target=labels.view(batch_size * input_seq_len))
                final_loss = lm_loss
            _, _, final_topk_entity = self.EntityLinkHead(input_ids=input_ids,
                                                          hidden_state=upper_encoder_output,
                                                          entity_link=input_entity_link,
                                                          entity_mask=entity_mask,
                                                          topk=self.inference_nearest_k_entity,
                                                          last_entity_only=last_entity_only)

            topk_entity = (encoder_topk_entity, final_topk_entity)

        # TODO: There is a bug for past_key_values
        all_encoder_hidden_states = None, None
        if output_hidden_states:
            all_encoder_hidden_states = lower_encoder_output_class.hidden_states + \
                                        upper_encoder_output_class.hidden_states
        all_encoder_attentions = None
        if output_attentions:
            all_encoder_attentions = lower_encoder_output_class.attentions + \
                                     upper_encoder_output_class.attentions
        detail_loss = DetailLoss({
            "lm_loss": lm_loss,
            "el_loss": el_loss,
            "encoder_elloss": encoder_memory_loss,
            "decoder_elloss": torch.tensor(0.0),
            "final_elloss": final_memory_loss
        }).to(self.device)

        return EMAGModelOutput(
            loss=final_loss,
            detail_loss=detail_loss,
            logits=lm_logits,
            topk_entity=topk_entity,
            # past_key_values=upper_decoder_output_class.past_key_values,
            encoder_last_hidden_state=upper_encoder_output,
            encoder_hidden_states=all_encoder_hidden_states,
            encoder_attentions=all_encoder_attentions
        )
