"""
Compute accuracy for the entity linking setting.
Accuracy is computed as comparing the top-1 predicted entity ID with the SLING linked entity ID
"""

import argparse
import json
import numpy as np


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f'Read {len(result)} data from {path}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred', required=True, help='Path to prediction file.')
    parser.add_argument('-gold', required=True, help='Path to ground-truth file')
    args = parser.parse_args()

    # ground_truth: a list of dict: {"id", "question", "answer", "answer_entity"}
    if args.gold.endswith(".json"):
        ground_truth = json.load(open(args.gold, 'r', encoding='utf8'))
    elif args.gold.endswith(".jsonl"):
        ground_truth = read_jsonl_as_list(args.gold)
    else:
        raise ValueError("Invalid file type of args.gold!")
    print(f'Load {len(ground_truth)} ground-truth data from {args.gold}.')
    # pred_dict: a dict of {id: [answer, topk_entity_list]}
    pred_dict = json.load(open(args.pred, 'r', encoding='utf8'))
    print(f'Load {len(pred_dict)} predictions from {args.pred}.')

    # convert the ground-truth file to a dict format
    # {data_id: entity_id}
    gold_dict = {gt["id"]: gt["answer_entity"] if "answer_entity" in gt else None for gt in ground_truth}
    assert len(gold_dict) == len(pred_dict)

    ems = []
    linkable_ems = []
    for data_id, prediction in pred_dict.items():
        pred_top1 = prediction[1][0]
        gold_answer_list = gold_dict[data_id]
        # If non-linkable, directly identify the prediction as False
        if gold_answer_list is None:
            ems.append(False)
        elif len(gold_answer_list) == 1:
            gold_top1 = gold_answer_list[0][0]
            ems.append(pred_top1 == gold_top1)
            linkable_ems.append(pred_top1 == gold_top1)
        # If there are multiple answers, matching any of them is considered as correct
        else:
            gold_top1 = [answer[0] for answer in gold_answer_list]
            ems.append(pred_top1 in gold_top1)
            linkable_ems.append(pred_top1 in gold_top1)

    em_score = round(np.mean(ems) * 100, 2)  # convert to 100% scale, 2-digit rounding
    linkable_em_score = round(np.mean(linkable_ems) * 100, 2)
    print('Exact Match: ', em_score)
    print('Linkable EM: ', linkable_em_score)


if __name__ == "__main__":
    main()
