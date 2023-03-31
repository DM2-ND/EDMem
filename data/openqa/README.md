# OpenQA datasets in EDMem

### General format
Each file is in JSONL format, where each line corresponds to an example. Attributes of every example:

- `id`: index of the example
- `question`: open-domain question about real-world knowledge
- `answer`: all possible answers. They could be multiple different but correct answers to the question, or aliases to the same entity. Here, if the original answer is an alias to an entity name (entity names are collected from Wikipedia), we add the entity name to the answer list if it is not already present. This is to make a fair comparison with EaE[1], who used a similar approach according to our communication with EaE authors. Other major baselines that are implemented by us (BART, EncDec, EncMem) also used the same data files for evaluation for fair comparison. This is a similar idea to [2], which points out that not considering entity aliases is a common flaw in OpenQA evaluation.
- `link_offset`: the offset (in characters) of entity mentions in the question
- `link_length`: the length (in characters) of entity mentions in the question
- `link_target`: the index of the corresponding entity of this mention in our entity vocabulary
- `surface_name`: the surface name of the mention in the question
- `entity_name`: the name of the corresponding entity of this mention

The following attributes are present in the training set:
- `answer_link_offset`: the offset (in characters) of entity mentions in the answer
- `answer_link_length`: the length (in characters) of entity mentions in the answer
- `answer_link_target`: the index of the corresponding entity of this mention in our entity vocabulary
- `answer_surface_name`: the surface name of the mention in the answer
- `answer_entity_name`: the name of the corresponding entity of this mention
- `entity` (for TQA): the entity name of the answer. This attribute originally exists in the TQA dataset, while other attributes related to entities and mentions are collected by us.

[1] Fevry et al. Entities as experts: Sparse memory access with entity supervision. 2020.
[2] Si et al. What's in a Name? Answer Equivalence For Open-Domain Question Answering. 2021.

Besides, in the `entity_linking` directories, the `sling_invocab-*.json` files contain the linked entities of the original answers. These results are obtained through Google's [SLING](https://github.com/google/sling) phrase table. These files are used to determine whether a question has an entity answer or non-entity answer.

Notes:
1. For the official dev split of TQA, the dev data file and the test data file are the same.