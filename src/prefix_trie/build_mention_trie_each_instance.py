import re
import json
import argparse
import pickle
from tqdm import tqdm
from transformers import BartTokenizerFast
from Trie import Trie


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-el_result', required=True, help='Path to the entity vocabulary file')
    parser.add_argument('-entid2mention', default=None, help="Path to the entid2mention_1M.json file, "
                                                             "if mentions need to be added to the trie")
    parser.add_argument('-topk', type=int, default=5, help="Top-k entity linking candidate to consider")
    parser.add_argument('-output', help='Path to output file.')
    parser.add_argument('-lowercase', default=False, action='store_true', help='Whether to lowercase the entities')
    args = parser.parse_args()

    el_predictions = json.load(open(args.el_result, 'r', encoding='utf8'))
    entid2mention = json.load(open(args.entid2mention, 'r', encoding='utf8'))

    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    tokenizer.add_special_tokens({"additional_special_tokens": ["<E_s>", "<E_e>"]})
    entity_start_token_id = tokenizer.convert_tokens_to_ids('<E_s>')
    entity_end_token_id = tokenizer.convert_tokens_to_ids('<E_e>')

    trie_each_instance = {}
    total_mention_cnt = 0
    for data_id, predictions in tqdm(el_predictions.items()):
        entity_name, entid_list = predictions
        topk_entid = entid_list[:args.topk]

        # Get the mention list for all the top-k entities
        mention_list = []
        for entid in topk_entid:
            mentions = entid2mention[str(entid)]
            for j in range(len(mentions)):
                mentions[j] = '<s> ' + mentions[j]
                mentions[j] = re.sub(r'\(.+\)$', '', mentions[j]).strip()
                # Some downstream tasks are lower-cased, e.g., Open QA
                if args.lowercase:
                    mentions[j] = mentions[j].lower()
            mention_list.extend(mentions)

        total_mention_cnt += len(mention_list)

        # Build trie for this instance
        token_ids = tokenizer(mention_list).input_ids

        # Remove the two BOS token at the front, replace the EOS token with <E_e>
        for i in range(len(token_ids)):
            token_ids[i] = token_ids[i][1:]
            token_ids[i][0] = entity_start_token_id
            token_ids[i][-1] = entity_end_token_id

        trie = Trie(token_ids)
        trie_each_instance[data_id] = trie.trie_dict

    if args.output is not None:
        pickle.dump(trie_each_instance, open(args.output, 'wb'))
    print(f'Average mentions per instance: {total_mention_cnt / len(el_predictions):.2f}')


if __name__ == "__main__":
    main()
