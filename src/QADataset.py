"""
Data classes for OpenQA datasets (NQ, TQA, WQ).
@Date  : 03/07/2022
@Author: Zhihan Zhang
@mail  : zzhang23@nd.edu
@homepage: ytyz1307zzh.github.io
"""
import pdb

import torch
import json
import argparse
import numpy as np
from typing import List, Dict
from transformers import BartTokenizer
import logging
from copy import deepcopy
from WikiDataset import WikiDataset
from QA_utils import get_exact_match
from Constants import NIL_ENTITY
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


class QADataset(torch.utils.data.Dataset):
    """
    Dataset class for Wikipedia pre-training
    """
    def __init__(self,
                 args: argparse.Namespace,
                 dataset: List[Dict],
                 id2entity: Dict[int, str],
                 logger: logging.Logger,
                 tokenizer: BartTokenizer,
                 entity_mask: torch.Tensor = None,
                 is_training: bool = True):
        """
        :param dataset: the raw data, stored in a list of dicts
        :param id2entity: index to entity dictionary
        :param logger: logging module
        :param tokenizer: transformer tokenizer
        :param is_training: training set or not
        :param entity_mask: 0-1 mask for entities.
        """
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.logger = logger
        self.is_training = is_training
        self.id2entity = id2entity
        self.generate_target = args.generate_target
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length
        self.add_another_bos = args.add_another_bos
        self.do_lowercase = not args.do_cased
        self.entity_mask = entity_mask.tolist() if entity_mask is not None else None
        # self.already_tokenized = already_tokenized

        self.index2id = {i: d["id"] for i, d in enumerate(self.dataset)}
        self.id2index = {d["id"]: i for i, d in enumerate(self.dataset)}
        self.data_ids = [d["id"] for d in self.dataset]
        self.answers = [d["answer"] for d in self.dataset]
        self.dataid2answer = {self.data_ids[idx]: self.answers[idx] for idx in range(len(self.data_ids))}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):

        # If the dataset is already a tokenized one, directly find the index
        # if self.already_tokenized:
        #     return self.dataset[index]

        instance = deepcopy(self.dataset[index])
        data_id = instance["id"]
        question = instance["question"]
        answer_list = instance["answer"]
        answer_entity = instance.get("entity", None)
        question_hyperlinks = {
            "link_offset": instance["link_offset"],
            "link_length": instance["link_length"],
            "link_target": instance["link_target"]
        }

        # First, process the input part
        # If the question does not end with a question mark, we add one
        if not question.endswith("?"):
            question = question + "?"

        # Lowercase question and answer
        if self.do_lowercase:
            question = question.lower()

        question_input_ids, \
            question_entity_link, \
            question_mention_mask = self.process_entity_link(question,
                                                             question_hyperlinks,
                                                             max_seq_length=self.max_input_length)

        if self.generate_target == "entity_linking":
            return self.get_entity_linking_data(instance, question_input_ids, question_entity_link)

        # In inference mode, only return data on input side
        if not self.is_training:
            item = {
                "id": data_id,
                "input_ids": question_input_ids,
            }
            if self.generate_target == "qa_pair":
                item["decoder_input_ids"] = question_input_ids
            return item

        # In training mode, we also need to get the labels (answers) as well as their entity mentions
        else:
            all_answer_hyperlinks = [{
                "link_offset": instance["answer_link_offset"][i],
                "link_length": instance["answer_link_length"][i],
                "link_target": instance["answer_link_target"][i],
            } for i in range(len(instance["answer_link_offset"]))]

            # Get the answer. For TQA, we use the Wiki entity as ground-truth answer.
            # For other datasets, we randomly pick one from the answer list.
            if answer_entity is not None:
                answer = answer_entity
                assert len(all_answer_hyperlinks) == 1
                answer_hyperlinks = all_answer_hyperlinks[0]
            else:
                assert len(answer_list) == len(all_answer_hyperlinks)
                random_answer_id = np.random.choice(range(len(answer_list)))
                answer = answer_list[random_answer_id]
                answer_hyperlinks = all_answer_hyperlinks[random_answer_id]

            if self.do_lowercase:
                answer = answer.lower()

            answer_input_ids, \
                answer_entity_link, \
                answer_mention_mask = self.process_entity_link(answer,
                                                               answer_hyperlinks,
                                                               max_seq_length=self.max_output_length)

            item = {
                "id": data_id,
                "input_ids": question_input_ids,
                "input_entity_link": question_entity_link,
                "labels": answer_input_ids,
                "output_entity_link": answer_entity_link
            }

            if self.generate_target == "qa_pair":
                item["decoder_input_ids"] = question_input_ids[:-1] + answer_input_ids[2:]
                item["labels"] = [-100 for _ in question_input_ids[:-1]] + answer_input_ids[2:]
                item["output_entity_link"] = question_entity_link[:-1] + answer_entity_link[2:]

            return item

    def process_entity_link(self, text, hyperlinks, max_seq_length):

        # add another <s> at the beginning. This would make the first token also start with a Ġ
        if self.add_another_bos:
            text = self.tokenizer.bos_token + ' ' + text
            added_chars = len(self.tokenizer.bos_token) + 1  # added characters due to prepending <s>
            for link_i in range(len(hyperlinks["link_offset"])):
                hyperlinks["link_offset"][link_i] += added_chars

        token_list = self.tokenizer.tokenize(text)

        # convert the char-based link annotations to match with the tokens
        if len(hyperlinks["link_offset"]) == 0:
            entity_link_by_token = [0 for _ in range(len(token_list))]
        else:
            entity_link_by_token = WikiDataset.identify_entity_mention(text, token_list, hyperlinks, self.tokenizer)

        # For those entities that are masked, do not label them in the sentence, so they won't be trained
        if self.entity_mask is not None:
            entity_link_by_token = [x if self.entity_mask[int(x)] == 0 else 0 for x in entity_link_by_token]

        tokens_with_entity, entity_link_by_token, entity_mention_mask \
            = WikiDataset.add_entity_special_tokens(token_list,
                                                    entity_link_by_token,
                                                    entity_start_token=self.args.entity_start_token,
                                                    entity_end_token=self.args.entity_end_token)
        # encode the tokens to ids. It adds <s> and </s> tokens to the ends of sequence
        # meanwhile, truncate the sequence to max_seq_length. If this argument is not set in data_args,
        # then the predefined model max length is used instead.
        input_ids = self.tokenizer.encode(tokens_with_entity,
                                          padding=False,
                                          truncation=True,
                                          max_length=max_seq_length)

        # truncate entity_link_by_token and entity_mention_mask if necessary
        # max_seq_length - 2 is to reserve places for <s> and </s>
        if len(entity_link_by_token) > (max_seq_length - 2):
            entity_link_by_token = entity_link_by_token[:max_seq_length - 2]
        if len(entity_mention_mask) > (max_seq_length - 2):
            entity_mention_mask = entity_mention_mask[:max_seq_length - 2]

        # accordingly, add two NIL_ENTITY tokens to the ends of entity_link list
        entity_link_by_token = [NIL_ENTITY] + entity_link_by_token + [NIL_ENTITY]
        entity_link_by_token = list(map(int, entity_link_by_token))  # convert string entity ids to integers
        entity_mention_mask = [0] + entity_mention_mask + [0]
        assert len(input_ids) == len(entity_link_by_token) == len(entity_mention_mask)

        # If entity mask is applied, convert the gold entity IDs
        if self.entity_mask is not None:
            entity_link_by_token = [x - sum(self.entity_mask[:x]) for x in entity_link_by_token]

        return input_ids, entity_link_by_token, entity_mention_mask

    def evaluate(self, data_ids, predictions):
        # find ground-truth answer using data_id, then compare with the prediction
        assert len(predictions) == len(data_ids)
        ems = []
        for did in range(len(data_ids)):
            ems.append(get_exact_match(predictions[did], self.dataid2answer[data_ids[did]]))
        return ems

    def save_predictions(self, data_ids, predictions, save_path):
        prediction_dict = {id_: prediction for id_, prediction in zip(data_ids, predictions)}
        with open(save_path, "w") as f:
            json.dump(prediction_dict, f)
        self.logger.info("Saved prediction in {}".format(save_path))

    def get_entity_linking_data(self, instance, question_input_ids, question_entity_link):
        """
        Get data when we only do entity linking task.
        """
        # Append a <E_s> to the question, i.e., insert before the EOS token
        if self.args.prepend_question_in_decoder_input:
            decoder_input_ids = question_input_ids[:-1] + [self.args.entity_start_token_id, question_input_ids[-1]]
        # If the question should not be prepended, i.e., the decoder input is purely a <E_s> placeholder
        else:
            if self.add_another_bos:
                decoder_input_ids = question_input_ids[:2] + [self.args.entity_start_token_id, question_input_ids[-1]]
            else:
                decoder_input_ids = question_input_ids[:1] + [self.args.entity_start_token_id, question_input_ids[-1]]
        # In entity linking, training inputs and testing inputs are completely the same
        # The only difference is the existence of labels
        if not self.is_training:
            item = {
                "id": instance["id"],
                "input_ids": question_input_ids,
                "decoder_input_ids": decoder_input_ids,
            }
        else:
            # The target entity id (if there are multiple linked answers, randomly pick one)
            linked_answer_entities = instance["answer_entity"]
            if len(linked_answer_entities) == 1:
                answer_entity = linked_answer_entities[0][0]
            else:
                select_idx = np.random.choice(list(range(len(linked_answer_entities))))
                answer_entity = linked_answer_entities[select_idx][0]

            # Add the target entity id as one of the entity linking loss label
            if self.args.prepend_question_in_decoder_input:
                output_entity_link = question_entity_link[:-1] + [answer_entity, question_entity_link[-1]]
            # If the question should not be prepended, i.e., the decoder input is purely a <E_s> placeholder
            else:
                if self.add_another_bos:
                    output_entity_link = question_entity_link[:2] + [answer_entity, question_entity_link[-1]]
                else:
                    output_entity_link = question_entity_link[:1] + [answer_entity, question_entity_link[-1]]

            # if self.args.final_layer_loss == "all":
            #     labels = output_entity_link
            # elif self.args.final_layer_loss == "answer_only":
            #     labels = [0 for _ in range(question_entity_link)]
            #     labels[-2] = answer_entity

            item = {
                "id": instance["id"],
                "input_ids": question_input_ids,
                "input_entity_link": question_entity_link,
                "decoder_input_ids": decoder_input_ids,
                "output_entity_link": output_entity_link,
                "labels": decoder_input_ids  # Only added to align with "qa_pair" input format, not used in practice
            }
        return item


class QADataLoader(DataLoader):

    def __init__(self, args, dataset, collate_fn, is_training):
        if is_training and not args.debug:
            sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
            batch_size = args.train_batch_size
        elif is_training and args.debug:
            sampler = SequentialSampler(dataset) if args.local_rank == -1 else \
                DistributedSampler(dataset, shuffle=False)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset) if args.local_rank == -1 else \
                DistributedSampler(dataset, shuffle=False)
            batch_size = args.predict_batch_size
        super(QADataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn)
