"""
Constants used in the model and experiments
@Date  : 01/22/2022
@Author: Zhihan Zhang
@mail  : zzhang23@nd.edu
@homepage: ytyz1307zzh.github.io
"""

# size of entity vocabulary
ENTITY_VOCAB_SIZE = 1000010

# filler of positions that are not entities in entity link vectors
NIL_ENTITY = 0

# number of tokens added to the tokenizer vocabulary (to the end)
NUM_ADDED_VOCAB = 2

# surface names of entity_start_token and entity_end_token
ENTITY_START_TOKEN = "<E_s>"
ENTITY_END_TOKEN = "<E_e>"
