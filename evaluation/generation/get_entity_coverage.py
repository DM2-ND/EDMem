"""
Calculate the entity coverage of the generated output (reference: entities in the ground-truth)
Entity aliases (mentions) are also taken into account.
"""

import json
import pdb
import re
import argparse
from tqdm import tqdm
from typing import Dict, List
import spacy
nlp = spacy.load("en_core_web_sm")


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f'Read {len(result)} data from {path}')
    return result


def read_json(path: str):
    return json.load(open(path, 'r', encoding='utf8'))


def is_sublist(short_list: List, long_list: List):
    """
    Check whether short_list is a sublist of long_list
    """
    for i in range(len(long_list) - len(short_list) + 1):
        if long_list[i:i + len(short_list)] == short_list:
            return True

    return False


def tokenize_sentences(texts: List[str]):
    """Normalize and tokenize strings"""
    output_iter = nlp.pipe(texts, disable=["tagger", "parser", "ner", "lemmatizer", "textcat"])

    norm_sentences = []
    for doc in output_iter:
        tokens = [str(w).strip().lower() for w in doc]
        norm_sentences.append(tokens)
    return norm_sentences


def clean_data(prediction: str):
    prediction = re.sub(r' <E_s> | <E_e> ', '', prediction).strip()
    prediction = re.sub(r'<E_s> | <E_e>', '', prediction).strip()
    prediction = re.sub(r'<E_s>|<E_e>', '', prediction).strip()
    prediction = re.sub(r' {2,}', ' ', prediction)
    return prediction


def clean_entity(entity: str):
    entity = re.sub(r'\(.+\)$', '', entity).strip()
    return entity


def get_entity_coverage(
        output_entity_name: List[str],
        entity2mention: Dict,
        prediction: str
):

    # Remove duplicate entities
    output_entity_name_set = set(output_entity_name)

    mention_coverage = 0
    prediction_tokens = tokenize_sentences([prediction])[0]

    for entity_name in output_entity_name_set:
        mention_name_list = entity2mention[entity_name]

        mention_set = tokenize_sentences([clean_entity(m) for m in mention_name_list])

        mention_match = []
        for name in mention_set:
            if is_sublist(short_list=name, long_list=prediction_tokens):
                mention_match.append(True)
            else:
                mention_match.append(False)
        mention_coverage += max(mention_match)

    mention_coverage = mention_coverage / len(output_entity_name_set)

    return mention_coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, help='Path to the dataset')
    parser.add_argument('-pred', required=True, help='Path to the prediction results')
    parser.add_argument('-entity2mention', required=True, help='Path to entity2mention.json')
    parser.add_argument('-remove_input', default=False, action="store_true",
                        help="If specified, entities that appear in input are not considered")
    args = parser.parse_args()

    dataset = read_jsonl_as_list(args.dataset)
    prediction_dict = read_json(args.pred)
    entity2mention = read_json(args.entity2mention)
    # These coverage scores are macro average scores
    average_mention_coverage = 0.0
    valid_instances = 0

    for instance in tqdm(dataset):
        id_ = instance["id"]
        outputs = instance["output"]
        output_entity_name = instance["output_entity_name"]

        # If specified, remove the entities that already appeared in input
        if args.remove_input:
            input_entity_name = instance["entity_name"]
            if isinstance(outputs, str):
                output_entity_name = []
                for j in range(len(instance["output_entity_name"])):
                    if instance["output_entity_name"][j] not in input_entity_name:
                        output_entity_name.append(instance["output_entity_name"][j])

            elif isinstance(outputs, list):
                output_entity_name = [[] for _ in range(len(outputs))]
                for j in range(len(instance["output_entity_name"])):
                    for k in range(len(instance["output_entity_name"][j])):
                        if instance["output_entity_name"][j][k] not in input_entity_name:
                            output_entity_name[j].append(instance["output_entity_name"][j][k])

            else:
                raise TypeError(f"Variable 'outputs' has a wrong type {type(outputs)}")

        prediction = clean_data(prediction_dict[id_])

        if isinstance(outputs, str):
            # If the target output have no entity, skip this one
            if len(output_entity_name) == 0:
                continue
            mention_coverage = get_entity_coverage(
                output_entity_name=output_entity_name,
                entity2mention=entity2mention,
                prediction=prediction)
            valid_instances += 1

        elif isinstance(outputs, list):
            assert len(outputs) == len(output_entity_name)
            # If the target output have no entity, skip this one
            if max([len(entity_list) for entity_list in output_entity_name]) == 0:
                continue
            mention_coverage = 0.0
            for i in range(len(outputs)):
                if len(output_entity_name[i]) == 0:
                    continue
                local_mention_coverage = get_entity_coverage(
                    output_entity_name=output_entity_name[i],
                    entity2mention=entity2mention,
                    prediction=prediction
                )
                mention_coverage = max(mention_coverage, local_mention_coverage)
            valid_instances += 1

        else:
            raise TypeError(f"Variable 'outputs' has a wrong type {type(outputs)}")

        average_mention_coverage += mention_coverage

    average_mention_coverage = average_mention_coverage / valid_instances
    print(f'Valid instances: {valid_instances}')
    print(f'Average entity coverage (considering all mention names): '
          f'{average_mention_coverage * 100:.2f}%')


if __name__ == "__main__":
    main()
