import os
import json
from typing import List
from tqdm import tqdm
import datasets

logger = datasets.logging.get_logger(__name__)


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    logger.info(f'Read {len(result)} data from {path}')
    return result


def load_wiki_data_from_dir(directory: str):
    """
    Load Wiki data from a directory.
    However, the directory must contain
    """
    file_list = [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
    logger.info(f"File list: {file_list}")
    dataset = []
    for file_name in file_list:
        assert file_name.endswith('.jsonl')
        data_subset = read_jsonl_as_list(os.path.join(directory, file_name))
        dataset.extend(data_subset)
    return dataset


def load_wiki_data_from_file(file_name: str):
    """
    Load Wiki data from a single file.
    """
    assert file_name.endswith('.jsonl')
    return read_jsonl_as_list(file_name)


# def checksum_tokenizer(text: str, tokens: List[str]):
#     """
#     Check whether the number of characters after tokenization matches the original text
#     A `False` result may come from unicode characters
#     """
#     tokenized_sum = sum([len(token) for token in tokens])
#     text_sum = len(text)
#     return tokenized_sum == text_sum


def load_wiki_data(path: str):
    if os.path.isdir(path):
        dataset = load_wiki_data_from_dir(path)
    elif os.path.isfile(path):
        dataset = load_wiki_data_from_file(path)
    else:
        raise ValueError("Invalid value for data path!")

    logger.info(f"{len(dataset)} data read from {path}.")
    return dataset


def load_id2entity(path: str):
    if path.endswith(".json"):
        id2entityitemid = json.load(open(path, 'r', encoding='utf8'))
        id2entity = {k: v["entity"] for k, v in id2entityitemid.items()}
        return id2entity
    else:
        raise ValueError("File types other than json are not supported yet.")
