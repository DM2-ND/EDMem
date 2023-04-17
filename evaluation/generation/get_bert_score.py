import pdb

from bert_score import score
from typing import List, Dict
import argparse
import re
import json
import torch


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f'Read {len(result)} data from {path}')
    return result


def clean_whitespaces(sentence):
    if isinstance(sentence, str):
        sentence = re.sub(r' +', ' ', sentence.strip())
    else:
        sentence = [re.sub(r' +', ' ', s.strip()) for s in sentence]
    return sentence


def calculate_bert_score(args, dataid2predict, dataid2target):
    predictions = [clean_whitespaces(s) for s in dataid2predict.values()]
    ground_truth = [clean_whitespaces(s) for s in dataid2target.values()]  # can be a List[str] or List[List[str]]

    P, R, F1 = score(cands=predictions,
                     refs=ground_truth,
                     lang='en',
                     idf=args.use_idf,
                     batch_size=args.batch_size,
                     rescale_with_baseline=args.rescale,
                     verbose=True)
    print(F1[:100])
    average_score = torch.mean(F1).item()

    return average_score


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
    parser.add_argument('-use_idf', default=False, action='store_true',
                        help='If specified, IDF score will be considered in BERTScore computation.')
    parser.add_argument('-batch_size', default=64, type=int, help='Batch size for BERT model.')
    parser.add_argument('-rescale', default=False, action='store_true',
                        help='Rescale the score to a more large range.')
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
    for data_id, prediction in pred_dict.items():
        if isinstance(prediction, dict):
            prediction = prediction['output']
        prediction = re.sub(r' <E_s> | <E_e> ', '', prediction).strip()
        prediction = re.sub(r'<E_s> | <E_e>', '', prediction).strip()
        prediction = re.sub(r'<E_s>|<E_e>', '', prediction).strip()
        prediction = re.sub(r' {2,}', ' ', prediction)
        pred_dict[data_id] = prediction
    print(f'Load {len(pred_dict)} predictions from {args.pred}.')

    # convert the ground-truth file to a dict format
    gold_dict = {gt["id"]: gt["output"] if isinstance(gt["output"], list) else [gt["output"]] for gt in ground_truth}
    assert len(gold_dict) == len(pred_dict)

    # For CREAK dataset, remove the prefix in ground-truth sentences (and in predictions if necessary)
    if args.remove_creak_prefix:
        gold_dict = remove_creak_prefix_multi_reference(gold_dict)
        if args.remove_pred_prefix:
            pred_dict = remove_creak_prefix(pred_dict)

    # sort gold_dict and pred_dict according to keys
    gold_dict = {k: gold_dict[k] for k in sorted(gold_dict.keys())}
    pred_dict = {k: pred_dict[k] for k in sorted(pred_dict.keys())}

    average_score = calculate_bert_score(args, pred_dict, gold_dict)
    print(f'Average BERTScore: {average_score * 100:.2f}')


if __name__ == "__main__":
    main()

