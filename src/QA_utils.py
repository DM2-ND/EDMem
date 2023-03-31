import pdb
import random
import torch.distributed as dist
from typing import List, Union
from Trie import Trie
import unicodedata
import re
import os
import json
import math
import torch
import numpy as np
import string


def mean(array: List):
    return sum(array) / len(array)


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


def mean_reduce(x, args):
    if args.n_gpu > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.divide_(args.n_gpu)
    return x


def sum_reduce(x, args):
    if args.n_gpu > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def combine_prediction_files(output_dir, n_gpu, prefix):
    all_results = {}
    file_list = [os.path.join(output_dir, "{}predictions_ps{}.json".format(prefix, gpu_number))
                 for gpu_number in range(n_gpu)]
    for filename in file_list:
        result = json.load(open(filename, 'r', encoding='utf8'))
        print(f'Loaded {len(result)} predictions from {filename}.')
        all_results.update(result)
    save_path = os.path.join(output_dir, "{}predictions.json".format(prefix))
    json.dump(all_results, open(save_path, 'w', encoding='utf8'), ensure_ascii=False)
    print(f'Saved {len(all_results)} predictions to {save_path}.')
    for filename in file_list:
        os.remove(filename)


def load_id2entity(path: str):
    if path.endswith(".json"):
        id2entityitemid = json.load(open(path, 'r', encoding='utf8'))
        id2entity = {int(k): v["entity"] for k, v in id2entityitemid.items()}
        return id2entity
    else:
        raise ValueError("File types other than json are not supported yet.")


def load_id2entitytokens(path: str):
    if path.endswith(".json"):
        id2entitytokens = json.load(open(path, 'r', encoding='utf8'))
        id2entitytokens = {int(k): v for k, v in id2entitytokens.items()}
        return id2entitytokens
    else:
        raise ValueError("File types other than json are not supported yet.")


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f'Read {len(result)} data from {path}')
    return result


def load_dataset(data_dir: str, task_name: str, datafile_prefix: str, rename_id=False):
    task_dir = os.path.join(data_dir, task_name)
    assert os.path.isdir(task_dir)

    train_set = read_jsonl_as_list(os.path.join(task_dir, datafile_prefix + task_name + '-train.jsonl'))
    random.shuffle(train_set)
    dev_set = read_jsonl_as_list(os.path.join(task_dir, datafile_prefix + task_name + '-dev.jsonl'))
    test_set = read_jsonl_as_list(os.path.join(task_dir, datafile_prefix + task_name + '-test.jsonl'))

    if rename_id:
        for i in range(len(train_set)):
            train_set[i]["id"] = task_name + "_" + train_set[i]["id"]
        for i in range(len(dev_set)):
            dev_set[i]["id"] = task_name + "_" + dev_set[i]["id"]
        for i in range(len(test_set)):
            test_set[i]["id"] = task_name + "_" + test_set[i]["id"]

    return {
        'train': train_set,
        'dev': dev_set,
        'test': test_set
    }


def load_all_datasets(data_dir: str, datafile_prefix: str):
    nq_data = load_dataset(data_dir, 'nq', datafile_prefix, rename_id=True)
    tqa_data = load_dataset(data_dir, 'tqa', datafile_prefix, rename_id=True)
    wq_data = load_dataset(data_dir, 'wq', datafile_prefix, rename_id=True)

    train_set = nq_data["train"] + tqa_data["train"] + wq_data["train"]
    random.shuffle(train_set)
    # dev_set = nq_data["dev"] + tqa_data["dev"] + wq_data["dev"]
    # test_set = nq_data["test"] + tqa_data["test"] + wq_data["test"]

    return {
        'train': train_set,
        'nq_dev': nq_data["dev"],
        'tqa_dev': tqa_data["dev"],
        'wq_dev': wq_data['dev'],
        'nq_test': nq_data['test'],
        'tqa_test': tqa_data['test'],
        'wq_test': wq_data['test']
    }


def calculate_num_steps(args, data_size: int):
    total_train_batch_size = args.train_batch_size
    if args.n_gpu > 0:
        total_train_batch_size *= args.n_gpu
        data_size = math.ceil(data_size / args.n_gpu) * args.n_gpu
        print("Data size for each epoch under DistributedSampler: ", data_size)
    total_epochs = args.num_train_epochs
    # if gradient_accumulation_steps == 1, then total_local_steps == total_global_steps
    total_local_steps = math.ceil(data_size / total_train_batch_size) * total_epochs
    total_global_steps = math.ceil(total_local_steps / args.gradient_accumulation_steps)
    warmup_steps = round(total_global_steps * args.warmup_ratio)
    return {
        "total_local_steps": total_local_steps,
        "total_global_steps": total_global_steps,
        "warmup_steps": warmup_steps
    }


def prepare_inputs(batch, device):
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device)
    if "labels" in batch.keys():
        batch["labels"] = batch["labels"].to(device)
        batch["input_entity_link"] = batch["input_entity_link"].to(device)
        batch["output_entity_link"] = batch["output_entity_link"].to(device)
    if "decoder_input_ids" in batch.keys():
        batch["decoder_input_ids"] = batch["decoder_input_ids"].to(device)
        batch["decoder_attention_mask"] = batch["decoder_attention_mask"].to(device)
    return batch


def convert_autoencoder_inputs(input_ids: torch.Tensor, entity_link: torch.Tensor = None, tokenizer=None):
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)

    eos_mask = (input_ids == tokenizer.eos_token_id)
    converted_input_ids = input_ids[~eos_mask].view(batch_size, seq_len - 1)
    converted_input_ids = torch.cat([converted_input_ids[:, :-1].clone(),
                                     torch.full((batch_size, 1), tokenizer.eos_token_id, dtype=torch.long),
                                     converted_input_ids[:, -1:].clone()], dim=1)

    if entity_link is None:
        return converted_input_ids

    assert input_ids.size() == entity_link.size()
    # The first entity_link element of each sequence will always be 0, since it is either </s> or <pad>
    converted_entity_link = entity_link[:, 1:].clone()
    converted_entity_link = torch.cat([converted_entity_link[:, :-1].clone(),
                                       torch.zeros((batch_size, 1), dtype=torch.long),
                                       converted_entity_link[:, -1:].clone()], dim=1)

    return converted_input_ids, converted_entity_link


def get_prefix_allowed_tokens_fn(args, tokenizer, trie):
    entity_start_token_id = args.entity_start_token_id
    entity_end_token_id = args.entity_end_token_id
    vocab_tokens = [i for i in range(len(tokenizer)) if i != entity_end_token_id]

    def prefix_allowed_tokens_fn(data_ids, batch_id, sentence: torch.LongTensor):
        """
        Return a 1-D list of allowed tokens at next timestep
        """
        sentence = sentence.tolist()
        status = get_status(sentence)

        if status == "Regular":
            trie_out = vocab_tokens
        else:  # status == "Entity"
            entity_start_pos = find_entity_start(sentence)
            trie_out = trie.get(sentence[entity_start_pos:])

        return trie_out

    def get_status(sentence: List[int]):
        """
        Check whether the model is currently generating an entity
        """
        num_entity_start = sum(token == entity_start_token_id for token in sentence)
        num_entity_end = sum(token == entity_end_token_id for token in sentence)

        if num_entity_start == num_entity_end:
            return "Regular"
        elif num_entity_start == num_entity_end + 1:
            return "Entity"
        else:
            raise ValueError(f"Invalid relationship between the number of <E_s> and <E_e>. "
                             f"<E_s> : {num_entity_start}, <E_e>: {num_entity_end}. Sentence: {sentence}")

    def find_entity_start(sentence: List[int]):
        """
        If the model is current generating an entity, find the part the has already generated
        """
        for i in range(len(sentence) - 1, -1, -1):
            if sentence[i] == entity_start_token_id:
                return i
            if i == 0 or sentence[i] == entity_end_token_id:
                raise ValueError(f"Invalid sentence for status \"Entity\": {sentence}")

    return prefix_allowed_tokens_fn


def get_prefix_allowed_tokens_fn_each_instance(args, tokenizer, trie_all_instances):
    entity_start_token_id = args.entity_start_token_id
    entity_end_token_id = args.entity_end_token_id
    vocab_tokens = [i for i in range(len(tokenizer)) if i != entity_end_token_id]

    def prefix_allowed_tokens_fn(data_ids, batch_id, sentence: torch.LongTensor):
        """
        Return a 1-D list of allowed tokens at next timestep
        """
        sentence = sentence.tolist()
        status = get_status(sentence)

        if status == "Finished":
            trie_out = [tokenizer.pad_token_id]
        elif status == "Regular":
            trie_out = vocab_tokens
        else:  # status == "Entity"
            entity_start_pos = find_entity_start(sentence)
            data_id = data_ids[batch_id]  # Get the instance ID
            instance_trie = Trie.load_from_dict(trie_all_instances[data_id])
            trie_out = instance_trie.get(sentence[entity_start_pos:])

        # data_id = data_ids[batch_id]  # Get the instance ID
        # instance_trie = Trie.load_from_dict(trie_all_instances[data_id])
        # trie_out = instance_trie.get(sentence)

        return trie_out

    def get_status(sentence: List[int]):
        """
        Check whether the model is currently generating an entity
        """
        num_entity_start = sum(token == entity_start_token_id for token in sentence)
        num_entity_end = sum(token == entity_end_token_id for token in sentence)

        if sentence[-1] == tokenizer.pad_token_id:
            return "Finished"
        elif num_entity_start == num_entity_end:
            return "Regular"
        elif num_entity_start == num_entity_end + 1:
            return "Entity"
        else:
            raise ValueError(f"Invalid relationship between the number of <E_s> and <E_e>. "
                             f"<E_s> : {num_entity_start}, <E_e>: {num_entity_end}. Sentence: {sentence}")

    def find_entity_start(sentence: List[int]):
        """
        If the model is current generating an entity, find the part the has already generated
        """
        for i in range(len(sentence) - 1, -1, -1):
            if sentence[i] == entity_start_token_id:
                return i
            if i == 0 or sentence[i] == entity_end_token_id:
                raise ValueError(f"Invalid sentence for status \"Entity\": {sentence}")

    return prefix_allowed_tokens_fn


def get_beam_scores_at_timestep(model_output_scores, batch_size, beam_size):
    model_output_scores = model_output_scores.view(batch_size, beam_size, -1)
    model_output_scores = torch.topk(model_output_scores, k=10, dim=-1)

    beam_indices = model_output_scores.indices
    beam_values = model_output_scores.values

    all_results = []
    # beam search
    if beam_size > 1:
        for batch_i in range(batch_size):
            batch_result = []
            for beam_i in range(beam_size):
                score_dict = {beam_indices[batch_i][beam_i][i].item():
                                    round(beam_values[batch_i][beam_i][i].item(), 3)
                              for i in range(beam_indices.size(-1))}
                batch_result.append(score_dict)
            all_results.append(batch_result)
    # greedy search
    else:
        for batch_i in range(batch_size):
            score_dict = {beam_indices[batch_i][0][i].item():
                                round(beam_values[batch_i][0][i].item(), 3)
                          for i in range(beam_indices.size(-1))}
            all_results.append(score_dict)

    return all_results


def get_beam_scores(beam_scores, batch_size, beam_size):
    """
    Get beam scores for each instance in the batch
    beam_scores: for beam search, it is (timestep, batch_size, beam_size, vocab_size);
                 for greedy search, it is (timestep, batch_size, vocab_size)
    """
    batch_beam_scores = [[] for _ in range(batch_size)]  # (batch_size, timestep, beam_size, top-k)
    for timestep in range(len(beam_scores)):
        top_scores = get_beam_scores_at_timestep(beam_scores[timestep], batch_size, beam_size)  # (batch, beam, k)
        for batch_i in range(batch_size):
            batch_beam_scores[batch_i].append(top_scores[batch_i])
    return batch_beam_scores


def find_eos(output_ids, eos_token_id):
    for i in range(len(output_ids) - 1, -1, -1):
        if output_ids[i] == eos_token_id:
            return i
    raise ValueError(f"Did not find any EOS token in generated sequence {output_ids}!")
