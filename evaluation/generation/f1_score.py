from typing import List, Union
from collections import Counter
import unicodedata
import argparse
import re
import os
import json
import numpy as np
import string


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f'Read {len(result)} data from {path}')
    return result


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


def remove_creak_prefix_multi_reference(dataid2target):
    for data_id, target_list in dataid2target.items():
        new_target_list = []

        for target in target_list:
            if target.startswith("This is true because") or target.startswith("this is true because"):
                prefix_length = len("This is true because")
                target = target[prefix_length:].strip()
                new_target_list.append(target)
            elif target.startswith("This is false because") or target.startswith("this is false because"):
                prefix_length = len("This is false because")
                target = target[prefix_length:].strip()
                new_target_list.append(target)
            else:
                raise ValueError(f"Invalid prefix of the sentence: {target}")

        dataid2target[data_id] = new_target_list

    return dataid2target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred', required=True, help='Path to prediction file.')
    parser.add_argument('-gold', required=True, help='Path to ground-truth file')
    parser.add_argument('-remove_creak_prefix', default=False, action="store_true",
                        help="specify to remove creak prefix in ground-truth sentences")
    parser.add_argument('-remove_pred_prefix', default=False, action="store_true",
                        help="If True, means the prediction prefix also need to be removed (only effective on CREAK)")
    args = parser.parse_args()

    gold_path = args.gold.split('/')
    gold_path[-1] = gold_path[-1].lower()
    args.gold = '/'.join(gold_path)

    # ground_truth: a list of dict: {"id", "input", "output"}
    if args.gold.endswith(".json"):
        ground_truth = json.load(open(args.gold, 'r', encoding='utf8'))
    elif args.gold.endswith(".jsonl"):
        ground_truth = read_jsonl_as_list(args.gold)
    else:
        raise ValueError("Invalid file type of args.gold!")
    print(f'Load {len(ground_truth)} ground-truth data from {args.gold}.')

    # pred_dict: a dict of {id: output}
    pred_dict = json.load(open(args.pred, 'r', encoding='utf8'))
    print(f'Load {len(pred_dict)} predictions from {args.pred}.')

    # convert the ground-truth file to a dict format
    gold_dict = {gt["id"]: gt["output"] for gt in ground_truth}
    assert len(gold_dict) == len(pred_dict)

    # For CREAK dataset, remove the prefix in ground-truth sentences (and in predictions if necessary)
    if args.remove_creak_prefix:
        gold_dict = remove_creak_prefix_multi_reference(gold_dict)
        if args.remove_pred_prefix:
            pred_dict = remove_creak_prefix(pred_dict)

    normalized_f1 = 0
    for data_id, prediction in pred_dict.items():
        gold_output = gold_dict[data_id]
        prediction = re.sub(r'<E_s>|<E_e>|<pad>', '', prediction).strip()
        if isinstance(gold_output, str):
            local_f1 = f1_score(prediction, gold_output)
        # If this is a multi-reference scenario
        elif isinstance(gold_output, list):
            local_f1 = max([f1_score(prediction, gold) for gold in gold_output])
        else:
            raise TypeError(f"Invalid type for variable gold_output: {type(gold_output)}")
        normalized_f1 += local_f1

    normalized_f1 = round(normalized_f1 / len(pred_dict) * 100, 2)
    print(f'F1 score: {normalized_f1}')


if __name__ == "__main__":
    main()

