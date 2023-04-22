import numpy as np
from typing import List, Dict, Tuple
import torch


def recall_n(topk_entity: np.ndarray, label: int, n: int):
    if label in topk_entity[:n]:
        return 1
    else:
        return 0


def compute_recall_metrics(topk_entity, labels):
    assert len(topk_entity) == len(labels)
    k = topk_entity.shape[1]
    # if k != 100, need to change the value of 'n's
    assert k == 100
    device = topk_entity.device

    recall_100, recall_20, recall_5, recall_1 = 0, 0, 0, 0
    for i in range(len(labels)):
        recall_100 += recall_n(topk_entity[i], labels[i], n=100)
        recall_20 += recall_n(topk_entity[i], labels[i], n=20)
        recall_5 += recall_n(topk_entity[i], labels[i], n=5)
        recall_1 += recall_n(topk_entity[i], labels[i], n=1)

    if len(labels) > 0:
        recall_100 = round(recall_100 / len(labels), 4)
        recall_20 = round(recall_20 / len(labels), 4)
        recall_5 = round(recall_5 / len(labels), 4)
        recall_1 = round(recall_1 / len(labels), 4)
    else:
        recall_100, recall_20, recall_5, recall_1 = 0, 0, 0, 0

    return {
        "R@100": torch.tensor(recall_100).to(device),
        "R@20": torch.tensor(recall_20).to(device),
        "R@5": torch.tensor(recall_5).to(device),
        "R@1": torch.tensor(recall_1).to(device)
    }


def combine_recall_metrics(encoder_recall=None, decoder_recall=None, final_recall=None):
    """
    Combine the recall metric dicts to a single dict.
    """
    return_dict = {}

    if encoder_recall is not None:
        return_dict["encoder_R@100"] = encoder_recall["R@100"]
        return_dict["encoder_R@20"] = encoder_recall["R@20"]
        return_dict["encoder_R@5"] = encoder_recall["R@5"]
        return_dict["encoder_R@1"] = encoder_recall["R@1"]

    if decoder_recall is not None:
        return_dict["decoder_R@100"] = decoder_recall["R@100"]
        return_dict["decoder_R@20"] = decoder_recall["R@20"]
        return_dict["decoder_R@5"] = decoder_recall["R@5"]
        return_dict["decoder_R@1"] = decoder_recall["R@1"]

    if final_recall is not None:
        return_dict["final_R@100"] = final_recall["R@100"]
        return_dict["final_R@20"] = final_recall["R@20"]
        return_dict["final_R@5"] = final_recall["R@5"]
        return_dict["final_R@1"] = final_recall["R@1"]

    return return_dict


def metrics_to_tuple(metrics: Dict[str, torch.Tensor]):
    """
    For encoder-decoder model, metrics include recalls from encoder, decoder and final head
    For auto-encoder model, metrics include recalls from encoder and final head
    """
    assert len(metrics) == 13 or len(metrics) == 9

    if len(metrics) == 13:
        return (
            metrics["encoder_R@100"],
            metrics["encoder_R@20"],
            metrics["encoder_R@5"],
            metrics["encoder_R@1"],
            metrics["decoder_R@100"],
            metrics["decoder_R@20"],
            metrics["decoder_R@5"],
            metrics["decoder_R@1"],
            metrics["final_R@100"],
            metrics["final_R@20"],
            metrics["final_R@5"],
            metrics["final_R@1"],
            metrics["num_mentions"]
        )
    elif len(metrics) == 9:
        return (
            metrics["encoder_R@100"],
            metrics["encoder_R@20"],
            metrics["encoder_R@5"],
            metrics["encoder_R@1"],
            metrics["final_R@100"],
            metrics["final_R@20"],
            metrics["final_R@5"],
            metrics["final_R@1"],
            metrics["num_mentions"]
        )
    # No other possibilities


def tuple_to_metrics(metric_tuple: Tuple[torch.Tensor]):
    """
    For encoder-decoder model, metrics include recalls from encoder, decoder and final head
    For auto-encoder model, metrics include recalls from encoder and final head
    """
    assert len(metric_tuple) == 13 or len(metric_tuple) == 9

    if len(metric_tuple) == 13:
        return {
            "encoder_R@100": metric_tuple[0],
            "encoder_R@20": metric_tuple[1],
            "encoder_R@5": metric_tuple[2],
            "encoder_R@1": metric_tuple[3],
            "decoder_R@100": metric_tuple[4],
            "decoder_R@20": metric_tuple[5],
            "decoder_R@5": metric_tuple[6],
            "decoder_R@1": metric_tuple[7],
            "final_R@100": metric_tuple[8],
            "final_R@20": metric_tuple[9],
            "final_R@5": metric_tuple[10],
            "final_R@1": metric_tuple[11],
            "num_mentions": metric_tuple[12]
        }
    elif len(metric_tuple) == 9:
        return {
            "encoder_R@100": metric_tuple[0],
            "encoder_R@20": metric_tuple[1],
            "encoder_R@5": metric_tuple[2],
            "encoder_R@1": metric_tuple[3],
            "final_R@100": metric_tuple[4],
            "final_R@20": metric_tuple[5],
            "final_R@5": metric_tuple[6],
            "final_R@1": metric_tuple[7],
            "num_mentions": metric_tuple[8]
        }
    # No other possibilities


def zero_dim_concat(a: np.ndarray, b: np.ndarray):
    """
    Tensor a could be 1-dim or 0-dim, tensor b is 1-dim or 0-dim
    Three possibilities in total:
    (1) a is scalar, b is scalar: single device, first concat
    (2) a is list, b is scalar: single device, subsequent concat
    (3) a is list, b is list: distributed setting
    """
    assert a.ndim == 1 or a.ndim == 0
    assert b.ndim == 1 or b.ndim == 0  # 1-dim: distributed setting, 0-dim: single device

    if a.ndim == 0:
        assert b.ndim == 0
        return np.stack((a, b), axis=0)
    elif a.ndim == 1:
        if b.ndim == 1:
            return np.concatenate((a, b), axis=0)
        else:
            return np.concatenate((a, np.expand_dims(b, axis=0)), axis=0)


def metric_concat(metric_host: Dict, new_metrics: Dict):
    assert list(metric_host.keys()) == list(new_metrics.keys())
    for key in metric_host.keys():
        array = metric_host[key]
        new_array = new_metrics[key]
        if isinstance(array, dict):
            for subkey in array.keys():
                array[subkey] = zero_dim_concat(array[subkey], new_array[subkey])
            metric_host[key] = array
        elif isinstance(array, np.ndarray):
            array = zero_dim_concat(array, new_array)
            metric_host[key] = array
        else:
            raise TypeError("Invalid type!")
    return metric_host


def weighted_average_list(values: List[float], num_items: List[int]):
    """
    Calculate the weighted average of values, with weights given by the number of items
    """
    assert len(values) == len(num_items)
    total_items = 0
    sum_values = 0

    for value, items in zip(values, num_items):
        sum_values += value * items
        total_items += items

    average_value = sum_values / total_items
    return average_value


def weighted_average_metrics(metrics: Dict):
    """
    Calculate weighted average of each metric value based on the number of mentions in each batch
    """
    num_mentions = metrics["num_mentions"]
    for key in metrics.keys():
        value = metrics[key]
        if isinstance(value, dict):
            for subkey in value.keys():
                value[subkey] = weighted_average_list(value[subkey], num_mentions)
        elif isinstance(value, list):
            value = weighted_average_list(value, num_mentions)
        else:
            raise TypeError("Invalid type!")
        metrics[key] = value
    # put total number of mentions into metrics
    metrics["num_mentions"] = sum(num_mentions)
    return metrics


# metric_host = {
#     "eval_encoder_EL_recall": {
#         "R@100": torch.tensor(2.1),
#         "R@20": torch.tensor(2.2),
#         "R@5": torch.tensor(2.3),
#         "R@1": torch.tensor(2.4)
#     },
#     "eval_decoder_EL_recall": {
#         "R@100": torch.tensor(2.5),
#         "R@20": torch.tensor(2.6),
#         "R@5": torch.tensor(2.7),
#         "R@1": torch.tensor(2.8)
#     },
#     "eval_final_EL_recall": {
#         "R@100": torch.tensor(2.9),
#         "R@20": torch.tensor(2.0),
#         "R@5": torch.tensor(2.11),
#         "R@1": torch.tensor(2.12)
#     },
#     "num_mentions": torch.tensor(20)
# }
# new_metrics = {
#     "eval_encoder_EL_recall": {
#         "R@100": torch.tensor(3.1),
#         "R@20": torch.tensor(3.2),
#         "R@5": torch.tensor(3.3),
#         "R@1": torch.tensor(3.4)
#     },
#     "eval_decoder_EL_recall": {
#         "R@100": torch.tensor(3.5),
#         "R@20": torch.tensor(3.6),
#         "R@5": torch.tensor(3.7),
#         "R@1": torch.tensor(3.8)
#     },
#     "eval_final_EL_recall": {
#         "R@100": torch.tensor(3.9),
#         "R@20": torch.tensor(3.0),
#         "R@5": torch.tensor(3.11),
#         "R@1": torch.tensor(3.12)
#     },
#     "num_mentions": torch.tensor(30)
# }
# metric_host = metric_concat(metric_host, new_metrics)
# print(metric_host)
# metric_host = metric_concat(metric_host, new_metrics)
# print(metric_host)
