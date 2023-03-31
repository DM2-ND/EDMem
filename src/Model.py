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
from transformers import BartPretrainedModel, BartForConditionalGeneration, BartConfig
from transformers.utils import logging
from transformers.generation_utils import (
    GreedySearchOutput,
    SampleOutput,
    BeamSearchOutput,
    BeamSampleOutput,
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput
)
from transformers.modeling_outputs import ModelOutput, Seq2SeqModelOutput
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from Beam import BeamScorer, BeamSearchScorer
from Encoder import EMAGLowerEncoder, EMAGUpperEncoder
from Decoder import EMAGLowerDecoder, EMAGUpperDecoder
from EntityMemory import EntityMemory
from Output import EMAGModelOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
from Loss import DetailLoss
from Constants import NIL_ENTITY
from Modules import PrefixConstrainedLogitsProcessor
from Trie import Trie
from model_utils import shift_tokens_right, find_instance_last_entity, squeeze_beam_inputs
from generation_utils import flatten_list
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]

logger = logging.get_logger(__name__)


class EMAGModel(BartPretrainedModel):
    """
    The main architecture of the EMAG model.
    """
    def __init__(
            self,
            config,
            model_args,
            entity_embedding=None
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

        # initialize entity embeddings, either load from file or random initialized
        if entity_embedding is not None:
            assert entity_embedding.shape == (self.entity_vocab_size, model_args.entity_embed_size)
            with torch.no_grad():
                self.EntityEmbedding.weight = nn.Parameter(entity_embedding.transpose(0, 1))
            del entity_embedding
        else:
            self._init_weights(self.EntityEmbedding)

        # entity memory module
        self.EncoderEntityMemory = EntityMemory(config, model_args, self.EntityEmbedding)
        self.DecoderEntityMemory = EntityMemory(config, model_args, self.EntityEmbedding)
        self.EntityLinkHead = EntityMemory(config, model_args, self.EntityEmbedding, output_embedding=False)

        # if train from scratch, load the configs from huggingface's BART
        if model_args.train_from_scratch:
            pretrained_bart_model = BartForConditionalGeneration(config=config)
        else:
            assert model_args.model_name_or_path is not None
            pretrained_bart_model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

        # load the word embeddings
        self.shared = pretrained_bart_model.model.shared

        # encoder and decoder blocks
        self.LowerEncoder = EMAGLowerEncoder(config, model_args, embed_tokens=self.shared,
                                             pretrained_encoder=pretrained_bart_model.model.encoder)
        self.UpperEncoder = EMAGUpperEncoder(config, model_args,
                                             pretrained_encoder=pretrained_bart_model.model.encoder)
        self.LowerDecoder = EMAGLowerDecoder(config, model_args, embed_tokens=self.shared,
                                             pretrained_decoder=pretrained_bart_model.model.decoder)
        self.UpperDecoder = EMAGUpperDecoder(config, model_args,
                                             pretrained_decoder=pretrained_bart_model.model.decoder)
        self.lower_decoder_attn = model_args.lower_decoder_attn

        # output heads
        self.LMHead = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))

        # tie weights of LMHead and model.shared
        self.tie_weights()

        # layer norm modules after entity memory access and before upper encoder/decoder
        self.EncoderMemoryLayerNorm = nn.LayerNorm(config.d_model)
        self.DecoderMemoryLayerNorm = nn.LayerNorm(config.d_model)

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

        if self.model_args.no_repeat_ngram_size != self.config.no_repeat_ngram_size:
            self.config.no_repeat_ngram_size = self.model_args.no_repeat_ngram_size

        self.pad_token_id = config.pad_token_id

        # Initialize weights and apply final processing
        # self.post_init()

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
        self.LowerDecoder.embed_tokens = self.shared

    def get_encoder(self):
        return NotImplementedError("get_encoder is ambiguous.")

    def get_decoder(self):
        return NotImplementedError("get_decoder is ambiguous.")

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.LMHead

    def set_output_embeddings(self, new_embeddings):
        self.LMHead = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            input_entity_link=None,
            output_entity_link=None,
            entity_mention_mask=None,
            metadata=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            entity_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            encoder_only=False,
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
        :param output_entity_link: entity linking labels of the output sequence, (batch, out_seq_len)
        :param entity_mention_mask: mask of entity mentions, (num_mentions, ) (ONLY USED IN compute_metrics)
        :param metadata: meta data, like data id (NOT USED)
        :param encoder_only: if `True`, will only execute the encoder part and output the encoder hidden states.
            Used in generation mode.
        :param entity_mask: a 0-1 vector indicating which entities to mask. Masked entities will receive an attention
                            score of -inf (which lead to an attention weight of 0)
        """

        # -------------------- Adapted from huggingface ---------------------
        # If labels is not None, meaning we are doing teacher-forcing language modeling
        # then our decoder inputs come from the labels.
        # Otherwise, the decoder input ids should be passed as a parameter
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    input_ids=labels,
                    pad_token_id=self.config.pad_token_id,
                    decoder_start_token_id=self.config.decoder_start_token_id
                )
                output_entity_link = shift_tokens_right(
                    input_ids=output_entity_link,
                    pad_token_id=NIL_ENTITY,
                    decoder_start_token_id=NIL_ENTITY
                )
        else:
            assert encoder_only or (decoder_input_ids is not None)
            assert input_entity_link is None and output_entity_link is None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # -------------------- Adapted from huggingface ---------------------

        # input and output tensor dimensions
        if input_ids is not None:
            assert input_ids.shape == attention_mask.shape
            batch_size = input_ids.size(0)
            input_seq_len = input_ids.size(1)
        else:
            batch_size = decoder_input_ids.size(0)
        if labels is not None:
            assert input_ids.shape == input_entity_link.shape
            assert labels.shape == output_entity_link.shape
            output_seq_len = labels.size(1)
        elif not encoder_only:
            output_seq_len = decoder_input_ids.size(1)

        # top-k nearest entities at memory access
        encoder_topk_entity, decoder_topk_entity, final_topk_entity, topk_entity = None, None, None, None
        # entity linking losses at memory access
        encoder_memory_loss, decoder_memory_loss, final_memory_loss = None, None, None

        if encoder_outputs is None:
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

            if encoder_only:
                return {"lower_encoder_output": lower_encoder_output,
                        "upper_encoder_output": upper_encoder_output,
                        "encoder_topk_entity": encoder_topk_entity}
        else:
            assert not self.training
            lower_encoder_output = encoder_outputs["lower_encoder_output"]
            upper_encoder_output = encoder_outputs["upper_encoder_output"]

        # lower decoder
        # TODO: note that we can use lower_encoder_output or upper_encoder_output as
        #  encoder_hidden_states here for the lower decoder
        if self.lower_decoder_attn == "lower_encoder":
            encoder_states_for_lower_decoder = lower_encoder_output
        elif self.lower_decoder_attn == "upper_encoder":
            encoder_states_for_lower_decoder = upper_encoder_output
        else:
            raise ValueError(f"Invalid value for model_args.lower_decoder_attn: {self.lower_decoder_attn}")

        lower_decoder_output_class = self.LowerDecoder(input_ids=decoder_input_ids,
                                                       attention_mask=decoder_attention_mask,
                                                       encoder_hidden_states=encoder_states_for_lower_decoder,
                                                       encoder_attention_mask=attention_mask,
                                                       output_attentions=output_attentions,
                                                       output_hidden_states=output_hidden_states,
                                                       return_dict=True)
        lower_decoder_output = lower_decoder_output_class.last_hidden_state
        assert lower_decoder_output.shape == (batch_size, output_seq_len, self.d_model)

        # decoder needs to access entity memory
        if self.training:
            decoder_memory_loss, decoder_entity_embedding, _ = \
                self.DecoderEntityMemory(input_ids=decoder_input_ids,
                                         hidden_state=lower_decoder_output,
                                         entity_link=output_entity_link,
                                         entity_mask=entity_mask)
        else:
            _, decoder_entity_embedding, decoder_topk_entity = \
                self.DecoderEntityMemory(input_ids=decoder_input_ids,
                                         hidden_state=lower_decoder_output,
                                         entity_link=output_entity_link,
                                         entity_mask=entity_mask,
                                         topk=self.inference_nearest_k_entity)

        # add entity embeddings to token embeddings with a layer norm
        upper_decoder_input = self.DecoderMemoryLayerNorm(decoder_entity_embedding + lower_decoder_output)
        assert upper_decoder_input.shape == (batch_size, output_seq_len, self.d_model)

        upper_decoder_output_class = self.UpperDecoder(inputs_embeds=upper_decoder_input,
                                                       attention_mask=decoder_attention_mask,
                                                       encoder_hidden_states=upper_encoder_output,
                                                       encoder_attention_mask=attention_mask,
                                                       output_attentions=output_attentions,
                                                       output_hidden_states=output_hidden_states,
                                                       return_dict=True)
        upper_decoder_output = upper_decoder_output_class.last_hidden_state
        assert upper_decoder_output.shape == (batch_size, output_seq_len, self.d_model)

        # language modeling head
        lm_logits = self.LMHead(upper_decoder_output) + self.final_logits_bias
        assert lm_logits.shape == (batch_size, output_seq_len, self.shared.num_embeddings)

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
                self.EntityLinkHead(input_ids=decoder_input_ids,
                                    hidden_state=upper_decoder_output,
                                    entity_link=output_entity_link,
                                    entity_mask=entity_mask,
                                    last_entity_only=last_entity_only)

            # if we only calculate entity linking loss in the last layer, set the encoder and decoder losses to zero
            if self.do_entity_linking and self.apply_el_loss == "final":
                el_loss = final_memory_loss
            # otherwise, the entity linking loss is the average of encoder, decoder and final-layer losses
            else:
                el_loss = (encoder_memory_loss + decoder_memory_loss + final_memory_loss) / 3

            if self.do_entity_linking:
                final_loss = el_loss
            else:
                lm_loss = self.LMLoss(input=lm_logits.view(batch_size * output_seq_len, self.shared.num_embeddings),
                                      target=labels.view(batch_size * output_seq_len))
                # the final loss value is the weighted sum of lm_loss and el_loss
                final_loss = lm_loss + self.elloss_weight * el_loss

        else:
            # In evaluation mode, labels are available during pre-training, but are not available in fine-tuning
            if labels is not None and not self.do_entity_linking:
                lm_loss = self.LMLoss(input=lm_logits.view(batch_size * output_seq_len, self.shared.num_embeddings),
                                      target=labels.view(batch_size * output_seq_len))
                final_loss = lm_loss
            _, _, final_topk_entity = self.EntityLinkHead(input_ids=decoder_input_ids,
                                                          hidden_state=upper_decoder_output,
                                                          entity_link=output_entity_link,
                                                          entity_mask=entity_mask,
                                                          topk=self.inference_nearest_k_entity,
                                                          last_entity_only=last_entity_only)

            topk_entity = (encoder_topk_entity, decoder_topk_entity, final_topk_entity)

        # TODO: There is a bug for past_key_values
        all_decoder_hidden_states, all_encoder_hidden_states = None, None
        if output_hidden_states:
            all_decoder_hidden_states = lower_decoder_output_class.hidden_states + \
                                        upper_decoder_output_class.hidden_states
            all_encoder_hidden_states = lower_encoder_output_class.hidden_states + \
                                        upper_encoder_output_class.hidden_states
        all_encoder_attentions, all_decoder_attentions = None, None
        all_cross_attentions = None
        if output_attentions:
            all_encoder_attentions = lower_encoder_output_class.attentions + \
                                     upper_encoder_output_class.attentions
            all_decoder_attentions = lower_decoder_output_class.attentions + \
                                     upper_decoder_output_class.attentions
            all_cross_attentions = lower_decoder_output_class.cross_attentions + \
                                   upper_decoder_output_class.cross_attentions

        detail_loss = DetailLoss({
            "lm_loss": lm_loss,
            "el_loss": el_loss,
            "encoder_elloss": encoder_memory_loss,
            "decoder_elloss": decoder_memory_loss,
            "final_elloss": final_memory_loss
        }).to(self.device)

        return EMAGModelOutput(
            loss=final_loss,
            detail_loss=detail_loss,
            logits=lm_logits,
            topk_entity=topk_entity,
            # past_key_values=upper_decoder_output_class.past_key_values,
            decoder_last_hidden_state=upper_decoder_output,
            decoder_hidden_states=all_decoder_hidden_states,
            decoder_attentions=all_decoder_attentions,
            cross_attentions=all_cross_attentions,
            encoder_last_hidden_state=upper_encoder_output,
            encoder_hidden_states=all_encoder_hidden_states,
            encoder_attentions=all_encoder_attentions
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # During generation, decoder_attention_mask should have same length as decoder_input_ids
        if decoder_attention_mask is not None and decoder_input_ids.shape[1] != decoder_attention_mask.shape[1]:
            # Check this is indeed caused by new words being generated
            # assert decoder_input_ids.shape[1] != encoder_outputs["lower_encoder_output"].shape[1]
            decoder_attention_mask = (decoder_input_ids != self.config.pad_token_id).to(torch.long)

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            encoder_outputs = self(input_ids=inputs_tensor,
                                   attention_mask=model_kwargs["attention_mask"],
                                   entity_mask=model_kwargs["entity_mask"],
                                   encoder_only=True)
            model_kwargs["encoder_outputs"] = encoder_outputs
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:

        if model_kwargs is not None and "decoder_input_ids" in model_kwargs \
                and model_kwargs["decoder_input_ids"] is not None:
            return model_kwargs.pop("decoder_input_ids")
        else:
            if "decoder_input_ids" in model_kwargs and model_kwargs["decoder_input_ids"] is None:
                model_kwargs.pop("decoder_input_ids")
            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * decoder_start_token_id

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        decoder_attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        # if "token_type_ids" in model_kwargs:
        #     token_type_ids = model_kwargs["token_type_ids"]
        #     model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)
        if decoder_attention_mask is not None:
            model_kwargs["decoder_attention_mask"] = decoder_attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["lower_encoder_output"] = encoder_outputs["lower_encoder_output"].index_select(
                0, expanded_return_idx.to(encoder_outputs["lower_encoder_output"].device)
            )
            encoder_outputs["upper_encoder_output"] = encoder_outputs["upper_encoder_output"].index_select(
                0, expanded_return_idx.to(encoder_outputs["upper_encoder_output"].device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        # model_kwargs["input_ids"] = model_kwargs["input_ids"].index_select(0, expanded_return_idx)

        return input_ids, model_kwargs

    def _get_logits_processor(
        self,
        data_ids: List[str],
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        encoder_no_repeat_ngram_size: int,
        encoder_input_ids: torch.LongTensor,
        bad_words_ids: List[List[int]],
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        remove_invalid_values: bool,
        logits_processor: Optional[LogitsProcessorList],
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant
        [`LogitsProcessor`] instances used to modify the scores of the language model head.
        """
        processors = LogitsProcessorList()

        # init warp parameters
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        encoder_no_repeat_ngram_size = (
            encoder_no_repeat_ngram_size
            if encoder_no_repeat_ngram_size is not None
            else self.config.encoder_no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        min_length = min_length if min_length is not None else self.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        diversity_penalty = diversity_penalty if diversity_penalty is not None else self.config.diversity_penalty
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )
        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
        )
        # instantiate processors list

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if diversity_penalty is not None and diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
            if self.config.is_encoder_decoder:
                processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))
            else:
                raise ValueError(
                    "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
                )
        if bad_words_ids is not None:
            processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(PrefixConstrainedLogitsProcessor(self.model_args,
                                                               data_ids,
                                                               prefix_allowed_tokens_fn,
                                                               num_beams // num_beam_groups))
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        if remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        return processors

    @torch.no_grad()
    def generate(
        self,
        data_ids: List[str],
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        return_topk_entity: Optional[bool] = False,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        id2entitytokens: Optional[Dict] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name
        inside the [`PretrainedConfig`] of the model. The default values indicated are the default
        values of those config.

        Most of these parameters are explained in more detail in [this blog post](https://huggingface.co/blog/how-to-generate).

        Parameters:

            inputs (`torch.Tensor` of shape `(batch_size, sequence_length)`, `(batch_size, sequence_length, feature_dim)` or `(batch_size, num_channels, height, width)`, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models
                `inputs` should of in the format of `input_ids`. For encoder-decoder models *inputs* can
                represent any of `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            max_length (`int`, *optional*, defaults to `model.config.max_length`):
                The maximum length of the sequence to be generated.
            max_new_tokens (`int`, *optional*, defaults to None):
                The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
                `max_new_tokens` or `max_length` but not both, they serve the same purpose.
            min_length (`int`, *optional*, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            repetition_penalty (`float`, *optional*, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
                `decoder_input_ids`.
            bad_words_ids(`List[List[int]]`, *optional*):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use `tokenizer(bad_word, add_prefix_space=True).input_ids`.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            max_time(`float`, *optional*, defaults to None):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for
                tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
                shape as `input_ids` that masks the pad token. [What are attention masks?](../glossary#attention-mask)
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            use_cache: (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (`int`, *optional*, defaults to 1):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                 Custom logits processors that complement the default logits processors built from arguments and a
                 model's config. If a logit processor is passed that is already created with the arguments or a model's
                 config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
                 model's config. If a stopping criteria is passed that is already created with the arguments or a
                 model's config an error is thrown. This feature is intended for advanced users.
            output_attentions (`bool`, *optional*, defaults to *False*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to *False*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to *False*):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to *False*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            forced_bos_token_id (`int`, *optional*):
                The id of the token to force as the first generated token after the `decoder_start_token_id`.
                Useful for multilingual models like [mBART](../model_doc/mbart) where the first generated token
                needs to be the target language token.
            forced_eos_token_id (`int`, *optional*):
                The id of the token to force as the last generated token when `max_length` is reached.
            remove_invalid_values (`bool`, *optional*):
                Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
                crash. Note that using `remove_invalid_values` can slow down generation.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If the
                model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific
                kwargs should be prefixed with *decoder_*.

        Return:
            [`~file_utils.ModelOutput`] or `torch.LongTensor`: A
            [`~file_utils.ModelOutput`] (if `return_dict_in_generate=True` or when
            `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the
                possible [`~file_utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchDecoderOnlyOutput`],
                    - [`~generation_utils.SampleDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSearchDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~file_utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchEncoderDecoderOutput`],
                    - [`~generation_utils.SampleEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSearchEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSampleEncoderDecoderOutput`]

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> # do greedy decoding without providing a prompt
        >>> outputs = model.generate(max_length=40)
        >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> document = (
        ... "at least two people were killed in a suspected bomb attack on a passenger bus "
        ... "in the strife-torn southern philippines on monday , the military said."
        ... )
        >>> # encode input context
        >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
        >>> # generate 3 independent sequences using beam search decoding (5 beams)
        >>> # with T5 encoder-decoder model conditioned on short news article.
        >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> input_context = "The dog"
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        >>> # generate 3 candidates using sampling
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, num_return_sequences=3, do_sample=True)
        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("ctrl")
        >>> model = AutoModelForCausalLM.from_pretrained("ctrl")
        >>> # "Legal" is one of the control codes for ctrl
        >>> input_context = "Legal My neighbor is"
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
        >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> input_context = "My cute dog"
        >>> # get tokens of words that should not be generated
        >>> bad_words_ids = tokenizer(["idiot", "stupid", "shut up"], add_prefix_space=True).input_ids
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        >>> # generate sequences without allowing bad_words to be generated
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
        >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        ```"""
        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        if pad_token_id is None and eos_token_id is not None:
            # special case if pad_token_id is not defined
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        if model_kwargs.get("attention_mask", None) is None:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        if self.config.is_encoder_decoder:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids.shape[-1]
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # 6. determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            data_ids=data_ids,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            logits_processor=logits_processor,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )

        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                id2entitytokens=id2entitytokens,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                return_topk_entity=return_topk_entity,
                synced_gpus=synced_gpus,
                id2entitytokens=id2entitytokens,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * num_return_sequences,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        return_topk_entity: Optional[bool] = False,
        synced_gpus: Optional[bool] = None,
        id2entitytokens: Optional[Dict] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from
                [`LogitsProcessor`] used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from
                [`StoppingCriteria`] used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to *False*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to *False*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to *False*):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to *False*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If
                model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation_utilsBeamSearchDecoderOnlyOutput`],
            [`~generation_utils.BeamSearchEncoderDecoderOutput`] or obj:*torch.LongTensor*: A
            `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.BeamSearchDecoderOnlyOutput`] if
            `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
            [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...    AutoTokenizer,
        ...    AutoModelForSeq2SeqLM,
        ...    LogitsProcessorList,
        ...    MinLengthLogitsProcessor,
        ...    BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList([
        ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ... ])

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        num_instances = batch_size * num_beams
        current_entity = {instance_id: None for instance_id in range(num_instances)}  # Record the current generating entity (for copy)

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # Copy the predicted entity from entity linking head to the generated sequence
            if self.model_args.entity_copy > 0:
                topk_entity = outputs.topk_entity[2]  # (num_mentions, topk)

                if len(topk_entity) > 0:
                    _, last_entity_each_instance = \
                        find_instance_last_entity(input_ids, self.model_args.entity_start_token_id)
                    assert len(last_entity_each_instance) == num_instances

                    top_beam_entity = {instance_id: None for instance_id in range(num_instances)}
                    top_entity_tokens = {}
                    for instance_id in range(num_instances):
                        last_entity_idx = last_entity_each_instance[instance_id]
                        if last_entity_idx is not None:
                            candidate_entities = topk_entity[last_entity_idx][:self.model_args.entity_copy].tolist()
                            top_beam_entity[instance_id] = candidate_entities
                            entity_tokens = flatten_list([id2entitytokens[entity_id] for entity_id in candidate_entities])
                            top_entity_tokens[instance_id] = entity_tokens

                    if current_entity != top_beam_entity:
                        batch_trie = {instance_id: Trie(token_ids).trie_dict
                                      for instance_id, token_ids in top_entity_tokens.items()}
                        prefix_allowed_tokens_fn = self.get_copy_allowed_tokens_fn(batch_trie)
                        prefix_processor = PrefixConstrainedLogitsProcessor(
                            self.model_args,
                            data_ids=None,
                            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                            num_beams=num_beams,
                            use_instance_id=True
                        )

                        # replace the original PrefixConstrainedLogitsProcessor with the new one
                        for i in range(len(logits_processor)):
                            if isinstance(logits_processor[i], PrefixConstrainedLogitsProcessor):
                                logits_processor[i] = prefix_processor
                        current_entity = top_beam_entity

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids=input_ids,
                                                 scores=next_token_scores,
                                                 logits=next_token_logits)
            # if return_dict_in_generate and output_scores:
            #     scores += (next_token_scores,)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        topk_entity_each_instance = None
        if return_topk_entity:
            generate_sequence = sequence_outputs["sequences"]
            model_inputs = self.prepare_inputs_for_generation(generate_sequence, **model_kwargs)
            model_inputs = squeeze_beam_inputs(model_inputs, batch_size, num_beams)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            topk_entity_last_step = outputs.topk_entity[2][:, :5]  # only keep top 5 for each entity prediction
            entity_range, _ = find_instance_last_entity(generate_sequence, self.model_args.entity_start_token_id)
            topk_entity_each_instance = [topk_entity_last_step[start_idx:end_idx, :].tolist()
                                         if start_idx != end_idx else []
                                         for start_idx, end_idx in entity_range]

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    topk_entity=topk_entity_each_instance,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    topk_entity=topk_entity_each_instance,
                )
        else:
            return sequence_outputs["sequences"]

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        id2entitytokens: Optional[Dict] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using greedy decoding.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from
                [`LogitsProcessor`] used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from
                [`StoppingCriteria`] used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to *False*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to *False*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to *False*):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to *False*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the
                model. If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_utils.GreedySearchDecoderOnlyOutput`],
            [`~generation_utils.GreedySearchEncoderDecoderOutput`] or obj:*torch.LongTensor*: A
            `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.GreedySearchDecoderOnlyOutput`] if
            `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
            [`~generation_utils.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ... AutoTokenizer,
        ... AutoModelForCausalLM,
        ... LogitsProcessorList,
        ... MinLengthLogitsProcessor,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList([
        ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
        ... ])

        >>> outputs = model.greedy_search(input_ids, logits_processor=logits_processor)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only

        batch_size = input_ids.size(0)
        current_entity = {batch_id: None for batch_id in range(batch_size)}  # Record the current generating entity (for copy)

        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # Copy the predicted entity from entity linking head to the generated sequence
            num_entity_copy = self.model_args.entity_copy
            if num_entity_copy > 0:
                topk_entity = outputs.topk_entity[2]  # (num_mentions, topk)

                if len(topk_entity) > 0:
                    _, last_entity_each_instance = \
                        find_instance_last_entity(input_ids, self.model_args.entity_start_token_id)

                    assert len(last_entity_each_instance) == batch_size
                    top1_entity = {batch_id: topk_entity[last_entity_each_instance[batch_id]][:num_entity_copy].tolist()
                                   for batch_id in range(batch_size)
                                   if last_entity_each_instance[batch_id] is not None}
                    top1_entity_tokens = {batch_id: flatten_list([id2entitytokens[entity_id] for entity_id in candidate_entities])
                                          for batch_id, candidate_entities in top1_entity.items()}

                    if current_entity != top1_entity:
                        batch_trie = {batch_id: Trie(token_ids).trie_dict
                                      for batch_id, token_ids in top1_entity_tokens.items()}
                        prefix_allowed_tokens_fn = self.get_copy_allowed_tokens_fn(batch_trie)
                        prefix_processor = PrefixConstrainedLogitsProcessor(
                            self.model_args,
                            data_ids=None,
                            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                            num_beams=1
                        )

                        # replace the original PrefixConstrainedLogitsProcessor with the new one
                        for i in range(len(logits_processor)):
                            if isinstance(logits_processor[i], PrefixConstrainedLogitsProcessor):
                                logits_processor[i] = prefix_processor
                        current_entity = top1_entity

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:

                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids=input_ids,
                                                  scores=next_token_logits,
                                                  logits=next_token_logits)

            if return_dict_in_generate and output_scores:
                scores += (next_tokens_scores,)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def get_copy_allowed_tokens_fn(self, batch_trie):
        entity_start_token_id = self.model_args.entity_start_token_id
        entity_end_token_id = self.model_args.entity_end_token_id
        vocab_size = self.shared.weight.size(0)
        vocab_tokens = [i for i in range(vocab_size) if i != entity_end_token_id]

        def copy_allowed_tokens_fn(data_ids, batch_id, sentence: torch.LongTensor):
            """
            Return a 1-D list of allowed tokens at next timestep
            """
            sentence = sentence.tolist()
            status = get_status(sentence)

            if status == "Finished":
                trie_out = [self.pad_token_id]
            elif status == "Regular":
                trie_out = vocab_tokens
            else:  # status == "Entity"
                entity_start_pos = find_entity_start(sentence)
                instance_trie = Trie.load_from_dict(batch_trie[batch_id])
                trie_out = instance_trie.get(sentence[entity_start_pos:])

                # Actually, I have no idea of the cause of this bug, but this is a way to circumvent
                if trie_out == [] and self.model_args.rescale_logits == "norm":
                    trie_out = [entity_end_token_id]

            return trie_out

        def get_status(sentence: List[int]):
            """
            Check whether the model is currently generating an entity
            """
            num_entity_start = sum(token == entity_start_token_id for token in sentence)
            num_entity_end = sum(token == entity_end_token_id for token in sentence)

            if sentence[-1] == self.pad_token_id:
                return "Finished"
            elif num_entity_start == num_entity_end:
                return "Regular"
            elif num_entity_start == num_entity_end + 1:
                return "Entity"
            else:
                raise ValueError(f"Invalid relationship between the number of <E_s> and <E_e>. "
                                 f"<E_s> : {num_entity_start}, <E_e>: {num_entity_end}. Sentence: {sentence}")

        def find_entity_start(sentence: List[int]):
            """
            If the model is current generating an entity, find the part the has already generated
            """
            for i in range(len(sentence) - 1, -1, -1):
                if sentence[i] == entity_start_token_id:
                    return i
                if i == 0 or sentence[i] == entity_end_token_id:
                    raise ValueError(f"Invalid sentence for status \"Entity\": {sentence}")

        return copy_allowed_tokens_fn
