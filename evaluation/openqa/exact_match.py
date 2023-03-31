"""
Calculate the exact match score of Natural Questions dataset.
Prediction file format: Dict[qid, prediction]
@Date  : 12/19/2021
@Author: Zhihan Zhang
@mail  : zzhang23@nd.edu
@homepage: ytyz1307zzh.github.io
"""

import numpy as np
import argparse
import unicodedata
from typing import List, Union
import random
import re
import json
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


def get_exact_match(prediction: str, groundtruth: Union[str, List[str]]):
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return normalize_answer(prediction) == normalize_answer(groundtruth)


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


def pure_entity_name(prediction: str):
    if not (prediction.startswith("<E_s>") and prediction.endswith("<E_e>")):
        return False
    if prediction.count("<E_s>") != 1:
        return False
    if prediction.count("<E_e>") != 1:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred', required=True, help='Path to prediction file.')
    parser.add_argument('-gold', required=True, help='Path to ground-truth file')
    parser.add_argument('-ans_link', default=None, help='Path to the result of answer linking.')
    parser.add_argument('-entity_vocab', required=True, help='Path to the 1M entity vocabulary.')
    args = parser.parse_args()

    entity_vocab = json.load(open(args.entity_vocab, 'r', encoding='utf8'))
    for i in range(len(entity_vocab)):
        entity_vocab[i] = entity_vocab[i].lower()
        entity_vocab[i] = re.sub(r'\(.+\)$', '', entity_vocab[i]).strip()
    entity_vocab = set(entity_vocab)

    # ground_truth: a list of dict: {"id", "question", "answer"}
    if args.gold.endswith(".json"):
        ground_truth = json.load(open(args.gold, 'r', encoding='utf8'))
    elif args.gold.endswith(".jsonl"):
        ground_truth = read_jsonl_as_list(args.gold)
    else:
        raise ValueError("Invalid file type of args.gold!")
    print(f'Load {len(ground_truth)} ground-truth data from {args.gold}.')
    # pred_dict: a dict of {id: answer}
    pred_dict = json.load(open(args.pred, 'r', encoding='utf8'))
    print(f'Load {len(pred_dict)} predictions from {args.pred}.')

    # convert the ground-truth file to a dict format
    gold_dict = {gt["id"]: gt["answer"] for gt in ground_truth}
    assert len(gold_dict) == len(pred_dict)

    # compare the two dicts using common keys
    entity_predictions = 0  # How many predictions are pure entities
    non_entity_predictions = 0  # How many predictions are not pure entities
    entity_prediction_ems = []
    non_entity_prediction_ems = []
    ems = []  # the result of exact match comparision

    for id_ in pred_dict.keys():
        prediction = pred_dict[id_]
        if isinstance(prediction, dict):
            prediction = prediction["output"]
        # Entity linking output may be a list of (prediction, topk_entity)
        if isinstance(prediction, list):
            prediction = prediction[0]
            prediction = re.sub(r'\(.+\)$', '', prediction).strip()
        pure_entity_flag = pure_entity_name(prediction)
        prediction = re.sub(r'<E_s>|<E_e>|<pad>', '', prediction).strip()
        inst_em = get_exact_match(prediction, gold_dict[id_])
        ems.append(inst_em)

        if pure_entity_flag:
            entity_predictions += 1
            entity_prediction_ems.append(inst_em)
        else:
            non_entity_predictions += 1
            non_entity_prediction_ems.append(inst_em)

    em_score = round(np.mean(ems) * 100, 2)  # convert to 100% scale, 2-digit rounding
    entity_pred_em_score = round(np.mean(entity_prediction_ems) * 100, 2)
    non_entity_pred_em_score = round(np.mean(non_entity_prediction_ems) * 100, 2)
    print('Exact Match: ', em_score)
    print(f'{entity_predictions} predictions are pure entities, EM: {entity_pred_em_score}; '
          f'{non_entity_predictions} predictions are not entities, EM: {non_entity_pred_em_score}')

    if args.ans_link is not None:

        ans_link = json.load(open(args.ans_link, 'r', encoding='utf8'))
        link_ems, unlink_ems = [], []  # EM result for linkable and unlinkable answers
        pure_entity_count, in_vocab_entity = 0, 0
        not_pure_but_right_linkable = 0
        not_valid_but_right_pure = 0
        pure_ems, valid_ems = [], []  # EM score when prediciton is a valid entity name

        for id_ in pred_dict.keys():

            raw_prediction = pred_dict[id_]
            # Beam search output may be a dict
            if isinstance(raw_prediction, dict):
                raw_prediction = raw_prediction["output"]
            # Entity linking output may be a list of (prediction, topk_entity)
            if isinstance(raw_prediction, list):
                raw_prediction = raw_prediction[0]
                raw_prediction = re.sub(r'\(.+\)$', '', raw_prediction).strip()
            raw_prediction = re.sub(r"<pad>", "", raw_prediction).strip()
            prediction = re.sub(r'<E_s>|<E_e>', '', raw_prediction).strip()
            inst_em = get_exact_match(prediction, gold_dict[id_])

            if ans_link[id_] is not None:  # this is an entity-linkable answer
                if inst_em == 1 and not pure_entity_name(raw_prediction):
                    not_pure_but_right_linkable += 1

                if pure_entity_name(raw_prediction):
                    pure_entity_count += 1
                    pure_ems.append(inst_em)

                    if prediction in entity_vocab:
                        in_vocab_entity += 1
                        valid_ems.append(inst_em)

                    if prediction not in entity_vocab and inst_em == 1:
                        not_valid_but_right_pure += 1
                        # print(f'Prediction: {raw_prediction}, answer: {gold_dict[id_]}')

                link_ems.append(inst_em)

            else:
                unlink_ems.append(inst_em)

        link_em_score = round(np.mean(link_ems) * 100, 2)
        pure_em_score = round(np.mean(pure_ems) * 100, 2)
        valid_em_score = round(np.mean(valid_ems) * 100, 2)
        unlink_em_score = round(np.mean(unlink_ems) * 100, 2)
        print('Linkable QA Exact Match: ', link_em_score)
        print(f'Among linkables, the ratio of predictions that are pure entity names: {pure_entity_count}/{len(link_ems)}')
        print('Pure entity names Exact Match: ', pure_em_score)
        print('Not predicted as pure entities, but got it right: ', not_pure_but_right_linkable)
        print(f'Among pure entity names, the ratio of predictions that are valid entity names: '
              f'{in_vocab_entity}/{pure_entity_count}')
        print('Valid entity names Exact Match: ', valid_em_score)
        print('Not predicted as valid entity names, but got it right: ', not_valid_but_right_pure)
        print(f'Linkable questions: {len(link_ems)}')
        print('Non-linkable QA Exact Match: ', unlink_em_score)
        print(f'Non-linkable questions: {len(unlink_ems)}')


if __name__ == "__main__":
    main()
