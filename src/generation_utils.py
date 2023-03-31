"""
Utility functions for generation tasks
"""
import os
import sys
path = os.path.abspath(__file__)
sys.path.append('/'.join(path.split('/')[:-2]+["/evaluation"]))

import random
import unicodedata
import re
import json
import string
from collections import Counter
from typing import List, Dict
from spacy.lang.en import English as NlpEnglish
from rouge import Rouge


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f'Read {len(result)} data from {path}')
    return result


def flatten_list(array: List[List]):
    """
    Reduce 1 dimension of a nested array
    """
    new_array = []
    for sub_array in array:
        assert isinstance(sub_array, list)
        new_array.extend(sub_array)
    return new_array


def normalize_answer(s: str):
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# F1 score definition
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def load_dataset(data_dir: str, task_name: str, datafile_prefix: str):
    task_dir = os.path.join(data_dir, task_name)
    assert os.path.isdir(task_dir), f"Directory {task_dir} does not exist!"

    train_set = read_jsonl_as_list(os.path.join(task_dir, datafile_prefix + task_name.lower() + '-train.jsonl'))
    random.shuffle(train_set)
    dev_set = read_jsonl_as_list(os.path.join(task_dir, datafile_prefix + task_name.lower() + '-dev.jsonl'))
    test_set = read_jsonl_as_list(os.path.join(task_dir, datafile_prefix + task_name.lower() + '-test.jsonl'))

    return {
        'train': train_set,
        'dev': dev_set,
        'test': test_set
    }


def normalize_batch(p_iter, p_batch_size=1000):
    """Normalize and tokenize strings.

    Args:
    p_iter (iter): iter over strings to normalize and tokenize.
    p_batch_size (int): number of batches.

    Returns:
    iter: iter over normalized and tokenized string.
    """

    NLP = NlpEnglish(parser=False)

    output_iter = NLP.pipe(p_iter, batch_size=p_batch_size)

    norm_sentences = []
    for doc in output_iter:
        tokens = [str(w).strip().lower() for w in doc]
        norm_sentences.append(' '.join(tokens))
    return norm_sentences


def normalize_single_target(predictions: Dict[str, str]):
    data_ids = list(predictions.keys())
    all_candidates = list(predictions.values())
    all_candidates = normalize_batch(all_candidates)
    return {data_ids[i]: [all_candidates[i]] for i in range(len(data_ids))}


def normalize_list_of_targets(dataid2target: Dict[str, List[str]]):
    data_ids = list(dataid2target.keys())
    all_references = list(dataid2target.values())

    def flatten(answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    flat_all_references, metadata = flatten(all_references)
    flat_all_references = normalize_batch(flat_all_references)
    return {data_ids[i]: flat_all_references[metadata[i][0]:metadata[i][1]] for i in range(len(data_ids))}


def get_rouge(predictions, dataid2target, reference_as_list=True):
    norm_predictions = normalize_single_target(predictions)
    if reference_as_list:
        norm_references = normalize_list_of_targets(dataid2target)
    else:
        norm_references = normalize_single_target(dataid2target)

    all_scores = {}

    rouge = Rouge()
    rouge_1, rouge_2, rouge_L = 0, 0, 0
    for id_ in norm_predictions.keys():
        prediction = norm_predictions[id_]
        references = norm_references[id_]
        local_rouge_1, local_rouge_2, local_rouge_L = 0, 0, 0

        if len(prediction[0]) > 0:
            for ref in references:
                scores = rouge.get_scores(prediction[0], ref, avg=True)
                local_rouge_1 = max(local_rouge_1, scores["rouge-1"]["f"])
                local_rouge_2 = max(local_rouge_2, scores["rouge-2"]["f"])
                local_rouge_L = max(local_rouge_L, scores["rouge-l"]["f"])

        rouge_1 += local_rouge_1
        rouge_2 += local_rouge_2
        rouge_L += local_rouge_L

    rouge_1 = rouge_1 / len(norm_predictions)
    rouge_2 = rouge_2 / len(norm_predictions)
    rouge_L = rouge_L / len(norm_predictions)

    all_scores['rouge_1'] = rouge_1
    all_scores['rouge_2'] = rouge_2
    all_scores['rouge_l'] = rouge_L

    return all_scores


def get_creak_accuracy(dataid2predict, dataid2target):
    assert len(dataid2predict) == len(dataid2target)
    accuracy = 0

    for data_id in dataid2predict.keys():
        prediction = dataid2predict[data_id]
        ground_truth = dataid2target[data_id]
        assert ground_truth == 'true' or ground_truth == 'false'
        accuracy += int(prediction == ground_truth)

    return accuracy / len(dataid2predict)


def remove_creak_prefix(dataid2target):
    for data_id, target in dataid2target.items():
        if target.startswith("This is true because") or target.startswith("this is true because"):
            prefix_length = len("This is true because")
            target = target[prefix_length:].strip()
            dataid2target[data_id] = target
        elif target.startswith("This is false because") or target.startswith("this is false because"):
            prefix_length = len("This is false because")
            target = target[prefix_length:].strip()
            dataid2target[data_id] = target
        else:
            raise ValueError(f"Invalid prefix of the sentence: {target}")

    return dataid2target
