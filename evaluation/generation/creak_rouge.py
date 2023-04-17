"""
Calculate ROUGE scores. This script is specifically for CREAK.
"""

import os
import pdb
import sys
import platform

if platform.platform().startswith("Windows"):
    path = os.path.abspath(__file__)
    sys.path.append('\\'.join(path.split('\\')[:-2]))
else:
    path = os.path.abspath(__file__)
    sys.path.append('/'.join(path.split('/')[:-2]))

from typing import List, Dict
from collections import Counter
import unicodedata
import argparse
import re
import json
import numpy as np
import string
from rouge import Rouge
from spacy.lang.en import English as NlpEnglish


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f'Read {len(result)} data from {path}')
    return result


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
            metadata.append((len(new_answers), len(new_answers) + len(answer)))
            new_answers += answer
        return new_answers, metadata

    flat_all_references, metadata = flatten(all_references)
    flat_all_references = normalize_batch(flat_all_references)
    return {data_ids[i]: flat_all_references[metadata[i][0]:metadata[i][1]] for i in range(len(data_ids))}


def get_rouge(predictions, dataid2target, reference_as_list=False):
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
        if reference_as_list:
            local_rouge_1, local_rouge_2, local_rouge_L = 0, 0, 0
            for ref in references:
                scores = rouge.get_scores(prediction[0], ref, avg=True)
                local_rouge_1 = max(local_rouge_1, scores["rouge-1"]["f"])
                local_rouge_2 = max(local_rouge_2, scores["rouge-2"]["f"])
                local_rouge_L = max(local_rouge_L, scores["rouge-l"]["f"])
        else:
            scores = rouge.get_scores(prediction[0], references[0], avg=True)
            local_rouge_1, local_rouge_2, local_rouge_L = \
                scores["rouge-1"]["f"], scores["rouge-2"]["f"], scores["rouge-l"]["f"]

        rouge_1 += local_rouge_1
        rouge_2 += local_rouge_2
        rouge_L += local_rouge_L

    rouge_1 = rouge_1 / len(norm_predictions)
    rouge_2 = rouge_2 / len(norm_predictions)
    rouge_L = rouge_L / len(norm_predictions)

    all_scores['ROUGE_1'] = rouge_1
    all_scores['ROUGE_2'] = rouge_2
    all_scores['ROUGE_L'] = rouge_L

    return all_scores


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
    parser.add_argument('-remove_pred_prefix', default=False, action="store_true",
                        help='If specified, remove the prefix in prediction')
    parser.add_argument('-multi_ref', default=False, action="store_true",
                        help='Indicate there are multiple references.')
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
    for data_id, prediction in pred_dict.items():
        if isinstance(prediction, dict):
            prediction = prediction["output"]
        prediction = re.sub(r' <E_s> | <E_e> ', '', prediction).strip()
        prediction = re.sub(r'<E_s> | <E_e>', '', prediction).strip()
        prediction = re.sub(r'<E_s>|<E_e>', '', prediction).strip()
        prediction = re.sub(r' {2,}', ' ', prediction)
        pred_dict[data_id] = prediction
    print(f'Load {len(pred_dict)} predictions from {args.pred}.')

    if args.remove_pred_prefix:
        pred_dict = remove_creak_prefix(pred_dict)

    # convert the ground-truth file to a dict format
    gold_dict = {gt["id"]: gt["output"] for gt in ground_truth}
    if args.multi_ref:
        gold_dict = remove_creak_prefix_multi_reference(gold_dict)
    else:
        gold_dict = remove_creak_prefix(gold_dict)
    assert len(gold_dict) == len(pred_dict)

    all_scores = get_rouge(pred_dict, gold_dict, reference_as_list=args.multi_ref)
    for metric, result in all_scores.items():
        print(f"{metric:10}: {result * 100:.2f}")


if __name__ == "__main__":
    main()

