# NQ

```bash
INPUT_DIR="data"
DATA_PREFIX="nq/edmem_"
# OUTPUT_DIR is the directory of trained checkpoints
```

### Fine-tuning

```bash
GPUS=8
SEED=3518
CKPT_NAME="nq-greedy-attn-upper_batch256-wait30-warmup0.05-lr1e-5-el1.0-es1.0-dropout0.1-eval300-TQAinit"
# When training on NQ, we loaded the checkpoint from TQA official-dev fine-tuning
PRETRAIN_DIR="tqa-greedy-attn-upper_batch256-wait30-warmup0.05-lr1e-5-el1.0-es1.0-dropout0.1-eval300-ckpt770k-ssm0.5-mlm0.3/3518"

train_args="
    --data_dir ${INPUT_DIR}/openqa
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task nq
    --datafile_prefix ${DATA_PREFIX}
    --do_train
    --output_dir ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}
    --model_name facebook/bart-large
    --pretrain_ckpt ${OUTPUT_DIR}/${PRETRAIN_DIR}
    --entity_embed_size 256
    --dropout 0.1
    --elloss_weight 1.0
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
    --train_batch_size 32
    --predict_batch_size 32
    --learning_rate 1e-5
    --weight_decay 0.01
    --max_grad_norm 0.1
    --gradient_accumulation_steps 1
    --num_train_epochs 300
    --warmup_ratio 0.05
    --wait_step 30
    --eval_period 300
    --seed ${SEED}
"

train_cmd="python -m torch.distributed.launch --nproc_per_node=${GPUS} src/run_QA.py ${train_args}"
```
P.S. This is run on 8x V100 GPUs. So the effective batch size is 256. Hyper-parameter search is mainly conducted on `learning_rate`, `elloss_weight` and `dropout`.

### Checkpoint

Fine-tuned checkpoint: [Google Drive link](https://drive.google.com/file/d/1cayU73pfMCVZ-sXOZWbCdp0rDRBIGNUQ/view?usp=sharing)


### Free-Form Generation

```bash
SEED=3518
SAVE_PREFIX="freeform_"
CKPT_NAME="nq-greedy-attn-upper_batch256-wait30-warmup0.05-lr1e-5-el1.0-es1.0-dropout0.1-eval300-TQAinit"

predict_args="
    --data_dir ${INPUT_DIR}/openqa
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task nq
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

This will lead to an EM of 28.89, with entity answers EM 34.00 and non-entity answers EM 16.70

### Dynamic Entity Linking

```bash
SEED=3518
SAVE_PREFIX="dynamic_"
CKPT_NAME="nq-greedy-attn-upper_batch256-wait30-warmup0.05-lr1e-5-el1.0-es1.0-dropout0.1-eval300-TQAinit"

predict_args="
    --data_dir ${INPUT_DIR}/openqa
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task nq
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
    --test_entity_trie ${INPUT_DIR}/prefix_trie/nq/nq_top1_entity_trie.pkl
    --trie_for_each_instance True
    --entity_copy 3
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

This will lead to an EM of 27.59, with entity answers EM 33.22 and non-entity answers EM 14.17

### Static Entity Linking

```bash
SEED=3518
SAVE_PREFIX="static_"
CKPT_NAME="nq-greedy-attn-upper_batch256-wait30-warmup0.05-lr1e-5-el1.0-es1.0-dropout0.1-eval300-TQAinit"

predict_args="
    --data_dir ${INPUT_DIR}/openqa
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task nq
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
    --test_entity_trie ${INPUT_DIR}/prefix_trie/nq/nq_top1_entity_trie.pkl
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

This will lead to an EM of 29.94, with entity answers EM 37.11 and non-entity answers EM 12.85

### Evaluation

```bash
eval_args="
    -pred ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}/${SAVE_PREFIX}predictions.json
    -gold ${INPUT_DIR}/openqa/nq/${DATA_PREFIX}nq-test.jsonl
    -ans_link ${INPUT_DIR}/openqa/nq/entity_linking/sling_invocab-test.json
    -entity_vocab ${INPUT_DIR}/wikipedia/entity_1m.json
"

eval_cmd="python evaluation/openqa/exact_match.py ${eval_args}"
```
