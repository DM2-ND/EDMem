# WQ

```bash
INPUT_DIR="data"
DATA_PREFIX="edmem_"
# OUTPUT_DIR is the directory of trained checkpoints
```

### Fine-tuning

```bash
GPUS=8
SEED=3518
CKPT_NAME="wq-greedy-attn-upper_batch128-wait30-warmup0.05-lr5e-6-el0.5-es1.0-dropout0.1-eval50-TQAinit"
# When training on NQ, we loaded the checkpoint from TQA official-dev fine-tuning
PRETRAIN_DIR="tqa-greedy-attn-upper_batch256-wait30-warmup0.05-lr1e-5-el1.0-es1.0-dropout0.1-eval300-ckpt770k-ssm0.5-mlm0.3/3518"

train_args="
    --data_dir ${INPUT_DIR}/openqa
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task wq
    --datafile_prefix ${DATA_PREFIX}
    --do_train
    --output_dir ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}
    --model_name facebook/bart-large
    --pretrain_ckpt ${OUTPUT_DIR}/${PRETRAIN_DIR}
    --entity_embed_size 256
    --dropout 0.1
    --elloss_weight 0.5
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
    --train_batch_size 16
    --predict_batch_size 32
    --learning_rate 5e-6
    --weight_decay 0.01
    --max_grad_norm 0.1
    --gradient_accumulation_steps 1
    --num_train_epochs 1000
    --warmup_ratio 0.05
    --wait_step 30
    --eval_period 50
    --seed ${SEED}
"

train_cmd="python -m torch.distributed.launch --nproc_per_node=${GPUS} src/run_QA.py ${train_args}"
```
**P.S.** This is run on 8x V100 GPUs. So the effective batch size is 128. The batch size is smaller than NQ/TQA because WQ has a small training set. Hyper-parameter search is mainly conducted on `learning_rate`, `elloss_weight` and `dropout`.

### Checkpoint

Fine-tuned checkpoint: [Google Drive link](https://drive.google.com/file/d/1wsNPF0BXQP2oQCysKKK5E8PQkC_RPF4w/view?usp=sharing)


### Free-Form Generation

```bash
SEED=3518
SAVE_PREFIX="freeform_"
CKPT_NAME="wq-greedy-attn-upper_batch128-wait30-warmup0.05-lr5e-6-el0.5-es1.0-dropout0.1-eval50-TQAinit"

predict_args="
    --data_dir ${INPUT_DIR}/openqa
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task wq
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

This will lead to an EM of 36.56, with entity answers EM 40.21 and non-entity answers EM 7.86

### Dynamic Entity Linking

```bash
SEED=3518
SAVE_PREFIX="dynamic_"
CKPT_NAME="wq-greedy-attn-upper_batch128-wait30-warmup0.05-lr5e-6-el0.5-es1.0-dropout0.1-eval50-TQAinit"

predict_args="
    --data_dir ${INPUT_DIR}/openqa
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task wq
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
    --test_entity_trie ${INPUT_DIR}/prefix_trie/wq/wq_top1_entity_trie.pkl
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

This will lead to an EM of 39.71, with entity answers EM 44.32 and non-entity answers EM 3.49

### Static Entity Linking

```bash
SEED=3518
SAVE_PREFIX="static_"
CKPT_NAME="wq-greedy-attn-upper_batch128-wait30-warmup0.05-lr5e-6-el0.5-es1.0-dropout0.1-eval50-TQAinit"

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
    --test_entity_trie ${INPUT_DIR}/prefix_trie/wq/wq_top1_entity_trie.pkl
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

This will lead to an EM of 41.09, with entity answers EM 45.87 and non-entity answers EM 3.49

### Evaluation

```bash
eval_args="
    -pred ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}/${SAVE_PREFIX}predictions.json
    -gold ${INPUT_DIR}/openqa/wq/${DATA_PREFIX}wq-test.jsonl
    -ans_link ${INPUT_DIR}/openqa/wq/entity_linking/sling_invocab-test.json
    -entity_vocab ${INPUT_DIR}/wikipedia/entity_1m.json
"

eval_cmd="python evaluation/openqa/exact_match.py ${eval_args}"
```
