"""
Build a dummy prefix trie for dynamic copying in generation tasks.
"""

import json
import argparse
import pickle
from transformers import BartTokenizerFast
from Trie import Trie


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
    parser.add_argument('-dev_data', required=True, help='Path to the dev set')
    parser.add_argument('-test_data', required=True, help='Path to the test set')
    parser.add_argument('-dev_output', help='Output file of dev prefix trie')
    parser.add_argument('-test_output', help='Output file of test prefix trie')
    args = parser.parse_args()

    dev_data = read_jsonl_as_list(args.dev_data)
    test_data = read_jsonl_as_list(args.test_data)

    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    tokenizer.add_special_tokens({"additional_special_tokens": ["<E_s>", "<E_e>"]})
    entity_start_token_id = tokenizer.convert_tokens_to_ids('<E_s>')
    entity_end_token_id = tokenizer.convert_tokens_to_ids('<E_e>')

    trie_each_instance = {}
    for instance in dev_data:
        data_id = instance["id"]

        # Get the mention list for all the top-k entities
        entity_name = "dummy dummy"
        entity_name = '<s> ' + entity_name

        # Build trie for this instance
        token_ids = tokenizer([entity_name]).input_ids

        # Remove the two BOS token at the front, replace the EOS token with <E_e>
        for i in range(len(token_ids)):
            token_ids[i] = token_ids[i][1:]
            token_ids[i][0] = entity_start_token_id
            token_ids[i][-1] = entity_end_token_id

        trie = Trie(token_ids)
        trie_each_instance[data_id] = trie.trie_dict

    if args.dev_output is not None:
        pickle.dump(trie_each_instance, open(args.dev_output, 'wb'))

    trie_each_instance = {}
    for instance in test_data:
        data_id = instance["id"]

        # Get the mention list for all the top-k entities
        entity_name = "dummy dummy"
        entity_name = '<s> ' + entity_name

        # Build trie for this instance
        token_ids = tokenizer([entity_name]).input_ids

        # Remove the two BOS token at the front, replace the EOS token with <E_e>
        for i in range(len(token_ids)):
            token_ids[i] = token_ids[i][1:]
            token_ids[i][0] = entity_start_token_id
            token_ids[i][-1] = entity_end_token_id

        trie = Trie(token_ids)
        trie_each_instance[data_id] = trie.trie_dict

    if args.test_output is not None:
        pickle.dump(trie_each_instance, open(args.test_output, 'wb'))


if __name__ == "__main__":
    main()



