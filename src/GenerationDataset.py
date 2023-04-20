"""
Data classes for Generation datasets
@Date  : 04/29/2022
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
from generation_utils import (
    get_f1_score,
    get_rouge,
    get_creak_accuracy,
    remove_creak_prefix
)
from Constants import NIL_ENTITY
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


class GenerationDataset(torch.utils.data.Dataset):
    """
    Dataset class for Wikipedia pre-training
    """
    def __init__(self,
                 args: argparse.Namespace,
                 dataset: List[Dict],
                 id2entity: Dict[int, str],
                 logger: logging.Logger,
                 tokenizer: BartTokenizer,
                 is_training: bool):
        """
        :param dataset: the raw data, stored in a list of dicts
        :param id2entity: index to entity dictionary
        :param logger: logging module
        :param tokenizer: transformer tokenizer
        :param is_training: training set or not
        """
        self.args = args
        self.dataset = dataset
        self.task_name = args.task
        self.tokenizer = tokenizer
        self.logger = logger
        self.is_training = is_training
        self.id2entity = id2entity
        self.generate_target = args.generate_target
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length
        self.add_another_bos = args.add_another_bos
        self.do_lowercase = not args.do_cased
        # self.already_tokenized = already_tokenized

        self.index2id = {i: d["id"] for i, d in enumerate(self.dataset)}
        self.id2index = {d["id"]: i for i, d in enumerate(self.dataset)}
        self.data_ids = [d["id"] for d in self.dataset]
        self.targets = [d["output"] for d in self.dataset]
        self.dataid2target = {self.data_ids[idx]: self.targets[idx] for idx in range(len(self.data_ids))}

        # Get the task-specific evaluation metric
        if self.generate_target == "entity_linking":
            self.metric = "Accuracy"
        elif self.task_name == "WoW":
            self.metric = "F1"
        elif self.task_name in ["MSMARCO", "eli5"]:
            self.metric = "Rouge"
        elif self.task_name == "creak":
            # if self.generate_target == "prefix":
            self.metric = "Rouge"
            # elif self.generate_target == "target":
            #     self.metric = "Accuracy"
        else:
            raise NotImplementedError(f"Did not have metric for task {self.task_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):

        # If the dataset is already a tokenized one, directly find the index
        # if self.already_tokenized:
        #     return self.dataset[index]

        instance = deepcopy(self.dataset[index])
        data_id = instance["id"]
        source = instance["input"]
        target = instance["output"]
        source_hyperlinks = {
            "link_offset": instance["link_offset"],
            "link_length": instance["link_length"],
            "link_target": instance["link_target"]
        }

        prefix_ids = self.get_prefix(instance)  # already wrapped with two BOS and one EOS (assume add_another_bos)

        if self.generate_target == "entity_linking":
            return self.get_entity_linking_data(instance)

        # First, process the input part
        # Lowercase input if necessary
        if self.do_lowercase:
            source = source.lower()

        source_input_ids, \
            source_entity_link, \
            source_mention_mask = self.process_entity_link(source,
                                                           source_hyperlinks,
                                                           max_seq_length=self.max_input_length)

        # In inference mode, only return data on input side
        if not self.is_training:
            item = {
                "id": data_id,
                "input_ids": source_input_ids,
            }
            if self.generate_target == "prefix":
                item["decoder_input_ids"] = prefix_ids
            return item

        # In training mode, we also need to get the labels (answers) as well as their entity mentions
        else:

            if isinstance(target, list):
                assert len(target) == 1
                target = target[0]
                target_hyperlinks = {
                    "link_offset": instance["output_link_offset"][0],
                    "link_length": instance["output_link_length"][0],
                    "link_target": instance["output_link_target"][0]
                }
            elif isinstance(target, str):
                target_hyperlinks = {
                    "link_offset": instance["output_link_offset"],
                    "link_length": instance["output_link_length"],
                    "link_target": instance["output_link_target"]
                }

            if self.do_lowercase:
                target = target.lower()

            target_input_ids, \
                target_entity_link, \
                target_mention_mask = self.process_entity_link(target,
                                                               target_hyperlinks,
                                                               max_seq_length=self.max_output_length)

            item = {
                "id": data_id,
                "input_ids": source_input_ids,
                "input_entity_link": source_entity_link,
                "labels": target_input_ids,
                "output_entity_link": target_entity_link
            }

            if self.generate_target == "prefix":
                item["decoder_input_ids"] = target_input_ids
                if self.task_name == "creak":
                    # len(prefix_ids) - 1 is the prefix occupied length in the whole target sequence
                    item["labels"] = [-100] * (len(prefix_ids) - 1) + target_input_ids[len(prefix_ids)-1:]
                    # Make sure that no entity exists in CREAK prefix
                    assert all([x == 0 for x in item["output_entity_link"][:len(prefix_ids)-1]])
                else:
                    raise NotImplementedError(f"Task {self.task_name} does not need a prefix!")

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

        return input_ids, entity_link_by_token, entity_mention_mask

    def evaluate(self, data_ids, predictions):
        # find ground-truth answer using data_id, then compare with the prediction
        assert len(predictions) == len(data_ids)

        if self.metric == "F1":
            scores = []
            for did in range(len(data_ids)):
                scores.append(get_f1_score(predictions[did], self.dataid2target[data_ids[did]]))
            return scores

        elif self.metric == "Rouge":
            dataid2predict = {data_ids[did]: predictions[did] for did in range(len(data_ids))}
            dataid2target = {data_ids[did]: self.dataid2target[data_ids[did]] for did in range(len(data_ids))}

            if self.task_name in ["MSMARCO", "eli5"]:
                reference_as_list = True
            elif self.task_name == "creak":
                reference_as_list = True
                if self.generate_target == "prefix":
                    dataid2target = remove_creak_prefix(dataid2target)
            else:
                raise NotImplementedError(f"Rouge metrics are not used by task {self.task_name}")

            rouge_score = get_rouge(dataid2predict, dataid2target, reference_as_list=reference_as_list)
            return rouge_score["rouge_l"]

        # elif self.metric == "Accuracy":
        #     dataid2predict = {data_ids[did]: predictions[did] for did in range(len(data_ids))}
        #     dataid2target = {data_ids[did]: self.dataid2target[data_ids[did]] for did in range(len(data_ids))}
        #
        #     if self.task_name == "creak":
        #         accuracy = get_creak_accuracy(dataid2predict, dataid2target)
        #     else:
        #         raise NotImplementedError(f"Accuracy metric is not used by task {self.task_name}")
        #
        #     return accuracy

        else:
            raise NotImplementedError(f"No evaluation function for metric {self.metric}")

    def save_predictions(self, data_ids, predictions, save_path):
        prediction_dict = {id_: prediction for id_, prediction in zip(data_ids, predictions)}
        with open(save_path, "w") as f:
            json.dump(prediction_dict, f)
        self.logger.info("Saved prediction in {}".format(save_path))

    def get_prefix(self, instance):
        if self.generate_target != "prefix":
            return None

        if self.task_name == "creak":
            target = instance["output"]
            # target[0]: for all references, the prefix (statement is true/false) is the same
            prefix = ' '.join(target[0].split()[:4])
            if prefix == "This is true because":
                prefix_ids = [152, 16, 1528, 142]
            elif prefix == "This is false because":
                prefix_ids = [152, 16, 3950, 142]
            else:
                raise ValueError(f"Invalid prefix: {prefix}")

            if self.do_lowercase:
                prefix_ids[0] = 42  # "this"

        else:
            raise NotImplementedError(f"Task {self.task_name} does not need a prefix!")

        if self.add_another_bos:
            prefix_ids = [self.tokenizer.bos_token_id] * 2 + prefix_ids + [self.tokenizer.eos_token_id]
        else:
            prefix_ids = [self.tokenizer.bos_token_id] + prefix_ids + [self.tokenizer.eos_token_id]

        return prefix_ids

    def get_entity_linking_data(self, instance):
        """
        Get data when we only do entity linking task.
        """
        source = instance["input"]
        target = instance["output"]
        source_hyperlinks = {
            "link_offset": instance["link_offset"],
            "link_length": instance["link_length"],
            "link_target": instance["link_target"]
        }

        # First, process the input part
        # Lowercase input if necessary
        if self.do_lowercase:
            source = source.lower()

        source_input_ids, \
            source_entity_link, \
            source_mention_mask = self.process_entity_link(source,
                                                           source_hyperlinks,
                                                           max_seq_length=self.max_input_length)

        if isinstance(target, list):
            if self.is_training:
                assert len(target) == 1
            target = target[0]
            target_hyperlinks = {
                "link_offset": instance["output_link_offset"][0],
                "link_length": instance["output_link_length"][0],
                "link_target": instance["output_link_target"][0]
            }
        elif isinstance(target, str):
            target_hyperlinks = {
                "link_offset": instance["output_link_offset"],
                "link_length": instance["output_link_length"],
                "link_target": instance["output_link_target"]
            }

        if self.do_lowercase:
            target = target.lower()

        target_input_ids, \
            target_entity_link, \
            target_mention_mask = self.process_entity_link(target,
                                                           target_hyperlinks,
                                                           max_seq_length=self.max_output_length)

        item = {
            "id": instance["id"],
            "input_ids": source_input_ids,
            "input_entity_link": source_entity_link,
            "labels": target_input_ids,
            "output_entity_link": target_entity_link
        }

        return item


class GenerationDataloader(DataLoader):

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
        super(GenerationDataloader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn)
