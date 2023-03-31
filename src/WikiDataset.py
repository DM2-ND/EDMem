"""
Wikipedia pre-train dataset
@Date  : 02/04/2022
@Author: Zhihan Zhang
@mail  : zzhang23@nd.edu
@homepage: ytyz1307zzh.github.io
"""
import pdb

import torch
import logging
import argparse
from copy import deepcopy
from transformers import BartTokenizer
from typing import List, Dict
from Constants import NIL_ENTITY


class WikiDataset(torch.utils.data.Dataset):
    """
    Dataset class for Wikipedia pre-training
    """
    def __init__(self,
                 args: argparse.Namespace,
                 dataset: List[Dict],
                 id2entity: Dict[int, str],
                 logger: logging.Logger,
                 tokenizer: BartTokenizer,
                 already_tokenized: bool,
                 is_training: bool):
        """
        :param dataset: the raw data, stored in a list of dicts
        :param id2entity: index to entity dictionary
        :param logger: logging module
        :param tokenizer: transformer tokenizer
        :param already_tokenized: whether the `dataset` argument is a tokenized dataset
        :param is_training: training set or not
        """
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.logger = logger
        self.is_training = is_training
        self.id2entity = id2entity
        self.max_seq_length = args.max_seq_length
        self.add_another_bos = args.add_another_bos
        self.already_tokenized = already_tokenized

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):

        # If the dataset is already a tokenized one, directly find the index
        if self.already_tokenized:
            return self.dataset[index]

        instance = deepcopy(self.dataset[index])
        data_id = instance["id"]
        passage_text = instance["text"]
        hyperlinks = instance["hyperlinks"]

        # First, process the input part
        # add another <s> at the beginning. This would make the first token also start with a Ġ
        if self.add_another_bos:
            passage_text = self.tokenizer.bos_token + ' ' + passage_text
            added_chars = len(self.tokenizer.bos_token) + 1  # added characters due to prepending <s>
            for link_i in range(len(hyperlinks["link_offset"])):
                hyperlinks["link_offset"][link_i] += added_chars

        token_list = self.tokenizer.tokenize(passage_text)

        # convert the char-based link annotations to match with the tokens
        entity_link_by_token = self.identify_entity_mention(passage_text, token_list, hyperlinks, self.tokenizer)

        tokens_with_entity, entity_link_by_token, entity_mention_mask \
            = self.add_entity_special_tokens(token_list,
                                             entity_link_by_token,
                                             entity_start_token=self.args.entity_start_token,
                                             entity_end_token=self.args.entity_end_token)
        # encode the tokens to ids. It adds <s> and </s> tokens to the ends of sequence
        # meanwhile, truncate the sequence to max_seq_length. If this argument is not set in data_args,
        # then the predefined model max length is used instead.
        input_ids = self.tokenizer.encode(tokens_with_entity,
                                          padding=False,
                                          truncation=True,
                                          max_length=self.max_seq_length)

        # truncate entity_link_by_token and entity_mention_mask if necessary
        # self.max_seq_length - 2 is to reserve places for <s> and </s>
        if len(entity_link_by_token) > (self.max_seq_length - 2):
            entity_link_by_token = entity_link_by_token[:self.max_seq_length-2]
        if len(entity_mention_mask) > (self.max_seq_length - 2):
            entity_mention_mask = entity_mention_mask[:self.max_seq_length-2]

        # accordingly, add two NIL_ENTITY tokens to the ends of entity_link list
        entity_link_by_token = [NIL_ENTITY] + entity_link_by_token + [NIL_ENTITY]
        entity_link_by_token = list(map(int, entity_link_by_token))  # convert string entity ids to integers
        entity_mention_mask = [0] + entity_mention_mask + [0]
        assert len(input_ids) == len(entity_link_by_token) == len(entity_mention_mask)

        # The output of wiki pre-training is the same as input (LM objective).
        # So we don't need another round of processing.
        # Return the data item, and convert data fields to tensors
        metadata = {"id": data_id}
        item = {
            "metadata": metadata,
            "input_ids": input_ids,
            "attention_mask": [1 for _ in range(len(input_ids))],
            "labels": deepcopy(input_ids),
            "input_entity_link": entity_link_by_token,
            "output_entity_link": deepcopy(entity_link_by_token),
            "entity_mention_mask": entity_mention_mask
        }
        return item

    @staticmethod
    def add_entity_special_tokens(token_list: List[str],
                                  entity_link_by_token: List[int],
                                  entity_start_token: str,
                                  entity_end_token: str):
        """
        Add entity-related special tokens (<E_s>, <E_e>) to the tokenized sequence
        :param token_list: tokenized sequence
        :param entity_link_by_token: entity mapping sequence
        :param entity_start_token: <E_s>
        :param entity_end_token: <E_e>
        """
        assert len(token_list) == len(entity_link_by_token)
        entity_start_token = entity_start_token
        entity_end_token = entity_end_token
        entity_start_placeholder = -1
        entity_end_placeholder = -2

        new_token_list = []
        new_entity_link_by_token = []
        for token_i in range(len(token_list)):
            current_label = entity_link_by_token[token_i]
            current_token = token_list[token_i]

            # handle boundary cases
            if token_i == 0:
                previous_label = NIL_ENTITY
            else:
                previous_label = entity_link_by_token[token_i - 1]

            # the current token is a entity mention
            if current_label != NIL_ENTITY:

                # in the middle of a mention, directly copy
                if previous_label == current_label:
                    new_entity_link_by_token.append(current_label)
                    new_token_list.append(current_token)
                # otherwise, it is the start of a new mention
                # if previous token is not a mention, just insert a <E_s>
                elif previous_label == NIL_ENTITY:
                    new_entity_link_by_token.append(entity_start_placeholder)
                    new_entity_link_by_token.append(current_label)
                    new_token_list.append(entity_start_token)
                    new_token_list.append(current_token)
                # if previous token is another mention, insert a <E_e> and a <E_s>
                else:
                    new_entity_link_by_token.append(entity_end_placeholder)
                    new_entity_link_by_token.append(entity_start_placeholder)
                    new_entity_link_by_token.append(current_label)
                    new_token_list.append(entity_end_token)
                    new_token_list.append(entity_start_token)
                    new_token_list.append(current_token)

                # if this is the last token, append a <E_s>
                if token_i == len(token_list) - 1:
                    new_entity_link_by_token.append(entity_end_placeholder)
                    new_token_list.append(entity_end_token)

            # if it is not a mention (current_label == NIL_ENTITY)
            else:
                # if previous token is the end of a mention, add a <E_e>
                if previous_label != NIL_ENTITY:
                    new_entity_link_by_token.append(entity_end_placeholder)
                    new_entity_link_by_token.append(current_label)
                    new_token_list.append(entity_end_token)
                    new_token_list.append(current_token)
                # otherwise, just copy
                else:
                    new_entity_link_by_token.append(current_label)
                    new_token_list.append(current_token)

        # get entity mention mask
        entity_mention_mask = []
        for token_i in range(len(new_entity_link_by_token)):
            if new_entity_link_by_token[token_i] == entity_start_placeholder:
                entity_mention_mask.append(0)
            elif new_entity_link_by_token[token_i] == entity_end_placeholder:
                entity_mention_mask.append(0)
            elif new_entity_link_by_token[token_i] == NIL_ENTITY:
                entity_mention_mask.append(0)
            else:
                entity_mention_mask.append(1)

        # convert entity_link list to "only <E_s> tokens have entity links"
        assert len(new_token_list) == len(new_entity_link_by_token)
        for token_i in range(len(new_token_list)):
            # find a <E_s>, then copy the entity id to it (which should be at position token_i + 1)
            if new_token_list[token_i] == entity_start_token:
                assert new_entity_link_by_token[token_i] == entity_start_placeholder
                new_entity_link_by_token[token_i] = new_entity_link_by_token[token_i + 1]
            # for other positions, directly set the entity_link value to NIL_ENTITY
            else:
                new_entity_link_by_token[token_i] = NIL_ENTITY

        return new_token_list, new_entity_link_by_token, entity_mention_mask

    @staticmethod
    def identify_entity_mention(text: str,
                                token_list: List[str],
                                hyperlinks: Dict[str, List],
                                tokenizer: BartTokenizer):
        """
        Use character-based hyperlink offsets and lengths to identify which tokens belong to a entity mention
        """
        link_offset = hyperlinks["link_offset"]
        link_length = hyperlinks["link_length"]
        link_target = hyperlinks["link_target"]

        current_offset = 0
        link_iter = 0
        entity_link_by_token = []  # indicates whether the current token is a part of a mention
        # the first mention in the passage, which is our next target
        next_link_offset = link_offset[link_iter]
        next_link_length = link_length[link_iter]
        next_link_target = link_target[link_iter]
        remain_length = 0  # the remaining characters of the current matched mention

        token_i = 0
        while token_i < len(token_list):
            token = token_list[token_i]
            # in rare cases, one unicode character in text can match multiple tokens after tokenization
            matched_tokens = 1

            if remain_length < 0:
                raise ValueError("Invalid mention boundary!")

            # if the token begins with Ġ, it indicates that there was a whitespace before tokenization.
            if token.startswith(chr(288)):
                # in rare cases, the link offset could be wrong (1-char shift to left)
                # we need to move it to the correct place by shifting one char to right
                if current_offset == next_link_offset:
                    next_link_offset += 1
                token = token[1:]  # remove Ġ when calculating its length
                current_offset += 1
                if remain_length == 1:
                    raise ValueError("Invalid mention boundary!")
                if remain_length > 0:
                    remain_length -= 1  # substract the whitespace from remaining length

            # if there is a mismatch in offset count between raw text and tokenized text
            if text[current_offset:current_offset+len(token)] != token:

                # in rare cases, this is caused the tokenizer did not use 'Ġ' to match a whitespace (I don't know why)
                if text[current_offset] == ' ':
                    current_offset += 1

                # if it is not the whitespace case
                if text[current_offset:current_offset + len(token)] != token:
                    # try to find the original token length in text
                    solved = False

                    def get_unicode_bpe(string: str):
                        return ''.join([tokenizer.byte_encoder[b] for b in string.encode("utf-8")])

                    while not solved:
                        for original_len in range(len(token)):
                            if get_unicode_bpe(text[current_offset:current_offset+original_len]) == token:
                                token = text[current_offset:current_offset+original_len]
                                solved = True
                                break
                        # if the bpe matching cannot find the solution, append next token and continue searching
                        if not solved:
                            token = token + token_list[token_i + 1]
                            token_i += 1
                            matched_tokens += 1

            # Finshing the last mention: duplicate the target id we just found
            if remain_length > 0:
                for _ in range(matched_tokens):
                    entity_link_by_token.append(entity_link_by_token[-1])
                current_offset += len(token)
                token_i += 1
                # In rare cases, hyperlink could only cover part of the token
                if remain_length < len(token):
                    remain_length = 0
                else:
                    remain_length -= len(token)
                # if finish this mention (multi-token mention), get the next target
                if remain_length == 0:
                    if link_iter != len(link_offset) - 1:
                        link_iter += 1
                    else:
                        continue
                    next_link_offset = link_offset[link_iter]
                    next_link_length = link_length[link_iter]
                    next_link_target = link_target[link_iter]
                continue

            # Looking for a new mention
            # In rare cases, there are some inconsistency between wiki hyperlink and BART tokenization
            # So a hyperlink can start from the middle from a BART token
            # (the tokenizer did not tokenize the sentence as wiki hyperlink expected)
            if current_offset < next_link_offset < current_offset + len(token):
                # adjust the length of the hyperlink
                next_link_length += next_link_offset - current_offset
                # manually set the hyperlink offset to the beginning of the current word
                next_link_offset = current_offset

            # if the current token not the start of an entity mention
            if current_offset != next_link_offset:
                for _ in range(matched_tokens):
                    entity_link_by_token.append(NIL_ENTITY)
                current_offset += len(token)
                token_i += 1
                continue

            # otherwise, we have find the start of a new entity mention
            for _ in range(matched_tokens):
                entity_link_by_token.append(next_link_target)
            current_offset += len(token)
            token_i += 1
            # In rare cases, hyperlink could only cover part of the token
            if next_link_length < len(token):
                remain_length = 0
            else:
                remain_length = next_link_length - len(token)

            # if finish this mention (single-token mention), get the next target
            if remain_length == 0:
                if link_iter != len(link_offset) - 1:
                    link_iter += 1
                else:
                    continue
                next_link_offset = link_offset[link_iter]
                next_link_length = link_length[link_iter]
                next_link_target = link_target[link_iter]

        return entity_link_by_token
