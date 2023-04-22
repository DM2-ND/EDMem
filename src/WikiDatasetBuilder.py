import os
import pdb
import time
from copy import deepcopy
from typing import Dict
import datasets
import random
from datasets.utils.download_manager import DownloadManager
from transformers import BartTokenizer
from WikiDataset import WikiDataset
from Constants import NIL_ENTITY
from data_utils import load_wiki_data

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "Wiki Dataset, all files stored in a directory"


class WikiPassageConfig(datasets.BuilderConfig):

    def __init__(
        self,
        tokenizer: BartTokenizer,
        max_seq_length: int,
        special_tokens: Dict[str, str],
        already_tokenized: bool,
        validation_split_percentage: float,
        add_another_bos: bool,
        **kwargs
    ):
        super(WikiPassageConfig, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.special_tokens = special_tokens
        self.already_tokenized = already_tokenized
        self.add_another_bos = add_another_bos
        self.validation_split_percentage = validation_split_percentage


class WikiPassageBuilder(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        WikiPassageConfig(
            name="default",
            tokenizer=None,
            max_seq_length=None,
            special_tokens=None,
            already_tokenized=None,
            validation_split_percentage=None,
            add_another_bos=None
        )
    ]

    # Two splits in total: train and eval
    split_names = [datasets.Split.TRAIN, datasets.Split.VALIDATION]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.features.Sequence(datasets.Value("int64")),
                    "input_entity_link": datasets.features.Sequence(datasets.Value("int64"))
                }
            )
        )

    def _split_generators(self, dl_manager: DownloadManager):
        if self.config.data_dir:
            assert os.path.isdir(self.config.data_dir)
        if self.config.data_files:
            assert len(self.config.data_files["train"]) == 1
            assert len(self.config.data_files["validation"]) == 1
            assert os.path.isfile(self.config.data_files["train"][0])
            assert os.path.isfile(self.config.data_files["validation"][0])

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "data_dir": self.config.data_dir,
                    "data_files": self.config.data_files,
                    "split_name": split,
                    "tokenizer": self.config.tokenizer,
                    "max_seq_length": self.config.max_seq_length,
                    "special_tokens": self.config.special_tokens,
                    "already_tokenized": self.config.already_tokenized,
                    "add_another_bos": self.config.add_another_bos,
                    "validation_split_percentage": self.config.validation_split_percentage
                }
            )
            for split in self.split_names
        ]

    def _generate_examples(
        self,
        data_dir: str,
        data_files: Dict[str, str],
        split_name: str,
        tokenizer: BartTokenizer,
        max_seq_length: int,
        special_tokens: Dict[str, str],
        already_tokenized: bool,
        add_another_bos: bool,
        validation_split_percentage: float
    ):
        """
        Args:
            data_dir - Directory containing data partition files. Each file should be a jsonl file.
            split_name - `train` or `validation`
            already_tokenized - If True, then the input files are already tokenized
            add_another_bos - Whether to add an additional BOS token at the front of each sequence
            validation_split_percentage - The portion of data to act as validation set
        """
        if data_dir is not None:

            dataset = load_wiki_data(data_dir)
            random.shuffle(dataset)
            # Size of training set
            train_size = round((100 - validation_split_percentage) / 100 * len(dataset))

            if split_name == "train":
                dataset = dataset[:train_size]
            elif split_name == "validation":
                dataset = dataset[train_size:]
            else:
                raise ValueError(f"Unsupported split_name {split_name}.")

        else:
            if split_name == "train":
                data_path = str(data_files["train"][0])
            elif split_name == "validation":
                data_path = str(data_files["validation"][0])
            else:
                raise ValueError(f"Unsupported split_name {split_name}.")

            dataset = load_wiki_data(data_path)

            # For training set, a shuffle is needed
            if split_name == "train":
                random.shuffle(dataset)

        # If the dataset is already a tokenized one, directly yield each item
        if already_tokenized:
            for item in dataset:
                data_id = item["id"]
                yield data_id, item

        else:
            entity_start_token = special_tokens["entity_start_token"]
            entity_end_token = special_tokens["entity_end_token"]

            # The logic should be the same with WikiDataset.__getitem__()
            for index in range(len(dataset)):

                instance = deepcopy(dataset[index])
                data_id = instance["id"]
                passage_text = instance["text"]
                hyperlinks = instance["hyperlinks"]

                # First, process the input part
                # add another <s> at the beginning. This would make the first token also start with a Ġ
                if add_another_bos:
                    passage_text = tokenizer.bos_token + ' ' + passage_text
                    added_chars = len(tokenizer.bos_token) + 1  # added characters due to prepending <s>
                    for link_i in range(len(hyperlinks["link_offset"])):
                        hyperlinks["link_offset"][link_i] += added_chars

                token_list = tokenizer.tokenize(passage_text)

                # convert the char-based link annotations to match with the tokens
                entity_link_by_token = WikiDataset.identify_entity_mention(passage_text, token_list,
                                                                           hyperlinks, tokenizer)

                tokens_with_entity, entity_link_by_token, entity_mention_mask \
                    = WikiDataset.add_entity_special_tokens(token_list,
                                                            entity_link_by_token,
                                                            entity_start_token=entity_start_token,
                                                            entity_end_token=entity_end_token)
                # encode the tokens to ids. It adds <s> and </s> tokens to the ends of sequence
                # meanwhile, truncate the sequence to max_seq_length. If this argument is not set in data_args,
                # then the predefined model max length is used instead.
                input_ids = tokenizer.encode(tokens_with_entity,
                                             padding=False,
                                             truncation=True,
                                             max_length=max_seq_length)

                # truncate entity_link_by_token and entity_mention_mask if necessary
                # self.max_seq_length - 2 is to reserve places for <s> and </s>
                if len(entity_link_by_token) > (max_seq_length - 2):
                    entity_link_by_token = entity_link_by_token[:max_seq_length - 2]
                if len(entity_mention_mask) > (max_seq_length - 2):
                    entity_mention_mask = entity_mention_mask[:max_seq_length - 2]

                # accordingly, add two NIL_ENTITY tokens to the ends of entity_link list
                entity_link_by_token = [NIL_ENTITY] + entity_link_by_token + [NIL_ENTITY]
                entity_link_by_token = list(map(int, entity_link_by_token))  # convert string entity ids to integers
                entity_mention_mask = [0] + entity_mention_mask + [0]
                assert len(input_ids) == len(entity_link_by_token) == len(entity_mention_mask)

                # The output of wiki pre-training is the same as input (LM objective).
                # So we don't need another round of processing.
                # Return the data item, and convert data fields to tensors
                item = {
                    "id": data_id,
                    "input_ids": input_ids,
                    "input_entity_link": entity_link_by_token
                }
                yield data_id, item
