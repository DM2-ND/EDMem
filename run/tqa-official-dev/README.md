# TQA In-house Test

```bash
INPUT_DIR="data"
DATA_PREFIX="tqa-official-dev/emag_"
# OUTPUT_DIR is the directory of trained checkpoints
```

### Free-Form Generation

```bash
SEED=3518
SAVE_PREFIX="freeform_"
CKPT_NAME="tqa-greedy-attn-upper_batch256-wait30-warmup0.05-lr1e-5-el1.0-es1.0-dropout0.1-eval300-ckpt770k-ssm0.5-mlm0.3"

predict_args="
    --data_dir ${INPUT_DIR}/openqa
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task tqa
    --datafile_prefix ${DATA_PREFIX}
    --do_predict
    --output_dir ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}
    --model_name facebook/bart-large
    --pretrain_ckpt None
    --entity_embed_size 256
    --entity_token_weight 1.0
    --lower_decoder_attn upper_encoder
    --inference_nearest_k_entity 100
    --generate_target answer
    --max_input_length 50
    --max_output_length 20
    --num_beams 1
    --do_sample False
    --no_repeat_ngram_size 3
    --add_another_bos True
    --predict_batch_size 64
    --prefix ${SAVE_PREFIX}
    --seed ${SEED}
"

predict_cmd="python -m torch.distributed.launch --nproc_per_node=${GPUS} src/run_QA.py ${predict_args}"
```

### Dynamic Entity Linking

```bash
SEED=3518
SAVE_PREFIX="dynamic_"
CKPT_NAME="tqa-greedy-attn-upper_batch256-wait30-warmup0.05-lr1e-5-el1.0-es1.0-dropout0.1-eval300-ckpt770k-ssm0.5-mlm0.3"

predict_args="
  --data_dir ${INPUT_DIR}/openqa
  --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
  --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
  --task tqa
  --datafile_prefix ${DATA_PREFIX}
  --do_predict
  --output_dir ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}
  --model_name facebook/bart-large
  --pretrain_ckpt None
  --entity_embed_size 256
  --entity_token_weight 1.0
  --lower_decoder_attn upper_encoder
  --inference_nearest_k_entity 100
  --generate_target answer
  --test_entity_trie ${INPUT_DIR}/prefix_trie/tqa/top1_entity_trie_each_instance_460.pkl
  --trie_for_each_instance True
  --entity_copy 1
  --rescale_logits none
  --max_input_length 50
  --max_output_length 20
  --num_beams 1
  --do_sample False
  --no_repeat_ngram_size 3
  --add_another_bos True
  --predict_batch_size 64
  --prefix ${SAVE_PREFIX}
  --seed ${SEED}
"

predict_cmd="python -m torch.distributed.launch --nproc_per_node=${GPUS} src/run_QA.py ${predict_args}"
```

This will lead to an EM of 44.96, with entity answers EM 51.89 and non-entity answers EM 7.38

### Static Entity Linking

```bash
SEED=3518
SAVE_PREFIX="static_"
CKPT_NAME="tqa-greedy-attn-upper_batch256-wait30-warmup0.05-lr1e-5-el1.0-es1.0-dropout0.1-eval300-ckpt770k-ssm0.5-mlm0.3"

predict_args="
  --data_dir ${INPUT_DIR}/openqa
  --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
  --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
  --task tqa
  --datafile_prefix ${DATA_PREFIX}
  --do_predict
  --output_dir ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}
  --model_name facebook/bart-large
  --pretrain_ckpt None
  --entity_embed_size 256
  --entity_token_weight 1.0
  --lower_decoder_attn upper_encoder
  --inference_nearest_k_entity 100
  --generate_target answer
  --test_entity_trie ${INPUT_DIR}/prefix_trie/tqa/top1_entity_trie_each_instance_460.pkl
  --trie_for_each_instance True
  --max_input_length 50
  --max_output_length 20
  --num_beams 1
  --do_sample False
  --no_repeat_ngram_size 3
  --add_another_bos True
  --predict_batch_size 64
  --prefix ${SAVE_PREFIX}
  --seed ${SEED}
"

predict_cmd="python -m torch.distributed.launch --nproc_per_node=${GPUS} src/run_QA.py ${predict_args}"
```

This will lead to an EM of 46.87, with entity answers EM 54.62 and non-entity answers EM 4.82

### Evaluation

```bash
eval_args="
    -pred ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}/${SAVE_PREFIX}predictions.json
    -gold ${INPUT_DIR}/openqa/${TASK}/${DATA_PREFIX}${TASK}-test.jsonl
    -ans_link ${INPUT_DIR}/openqa/${TASK}/tqa-official-dev/entity_linking/sling_invocab-test.json
    -entity_vocab ${INPUT_DIR}/wikipedia/entity_1m.json
"

eval_cmd="python evaluation/openqa/exact_match.py ${eval_args}"
```
