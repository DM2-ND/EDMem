# MSMARCO

```bash
INPUT_DIR="data"
DATA_PREFIX="edmem_"
# OUTPUT_DIR is the directory of trained checkpoints
```

### Fine-tuning & predict with free-form generation

```bash
GPUS=8
SEED=3518
CKPT_NAME="msmarco-batch256-wait20-warmup0.05-lr5e-6-el0.5-es1.0-dropout0.1-eval600"
PRETRAIN_DIR="scratch-attn_upper-gpu8-step1m-warmup0.1-batch2048-lr1e-4-norm0.1-ssm0.5-mlm0.3-el1.0/checkpoint-1000000"

SAVE_PREFIX="freeform_"
train_args="
    --data_dir ${INPUT_DIR}/generation
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task MSMARCO
    --datafile_prefix ${DATA_PREFIX}
    --do_train
    --do_predict
    --output_dir ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}
    --model_name facebook/bart-large
    --pretrain_ckpt ${OUTPUT_DIR}/${PRETRAIN_DIR}
    --entity_embed_size 256
    --dropout 0.1
    --elloss_weight 0.5
    --entity_token_weight 1.0
    --lower_decoder_attn upper_encoder
    --inference_nearest_k_entity 100
    --generate_target target
    --max_input_length 30
    --max_output_length 50
    --num_beams 5
    --do_sample False
    --no_repeat_ngram_size 3
    --add_another_bos True
    --train_batch_size 32
    --predict_batch_size 16
    --learning_rate 5e-6
    --weight_decay 0.01
    --max_grad_norm 0.1
    --gradient_accumulation_steps 1
    --num_train_epochs 300
    --warmup_ratio 0.05
    --wait_step 20
    --eval_period 600
    --seed ${SEED}
"

train_cmd="python -m torch.distributed.launch --nproc_per_node=${GPUS} src/run_generation.py ${train_args}"
```
P.S. This is run on 8x V100 GPUs. So the effective batch size is 256. Hyper-parameter search is mainly conducted on `learning_rate` (5e-6, 1e-5, 2e-5) and `elloss_weight` (0.5, 1.0, 2.0).

Free-form generation yields the following results on the test set: ROUGE_1 57.59, ROUGE_2 39.44, ROUGE_L 54.37, F1 55.00, BERTScore 89.40, entity coverage: total 45.32 / unseen 25.17.

### Checkpoint

Fine-tuned checkpoint: [Google Drive link](https://drive.google.com/file/d/1znbkGOXDnz-BPyfJFDnpE45wwDCcR_II/view?usp=sharing)

### Dynamic Entity Linking

```bash
SEED=3518
SAVE_PREFIX="dynamic_"
CKPT_NAME="msmarco-batch256-wait20-warmup0.05-lr5e-6-el0.5-es1.0-dropout0.1-eval600"

predict_args="
    --data_dir ${INPUT_DIR}/generation
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task MSMARCO
    --datafile_prefix ${DATA_PREFIX}
    --do_predict
    --output_dir ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}
    --model_name facebook/bart-large
    --pretrain_ckpt None
    --entity_embed_size 256
    --entity_token_weight 1.0
    --lower_decoder_attn upper_encoder
    --inference_nearest_k_entity 100
    --generate_target target
    --dev_entity_trie ${INPUT_DIR}/prefix_trie/MSMARCO/dummy_dev_trie.pkl
    --test_entity_trie ${INPUT_DIR}/prefix_trie/MSMARCO/dummy_test_trie.pkl
    --trie_for_each_instance True
    --entity_copy 5
    --rescale_logits norm
    --max_input_length 30
    --max_output_length 50
    --num_beams 5
    --do_sample False
    --no_repeat_ngram_size 10
    --add_another_bos True
    --predict_batch_size 16
    --prefix ${SAVE_PREFIX}
    --seed ${SEED}
"

predict_cmd="python -m torch.distributed.launch --nproc_per_node=${GPUS} src/run_generation.py ${predict_args}"
```

This will lead to the following results on the test set: ROUGE_1 55.68, ROUGE_2 37.07, ROUGE_L 52.59, F1 52.55, BERTScore 88.57, entity coverage: total 50.94 / unseen 28.31.

### Evaluation

ROUGE scores:
```bash
eval_args="
    -pred ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}/${SAVE_PREFIX}predictions.json
    -gold ${INPUT_DIR}/generation/MSMARCO/${DATA_PREFIX}msmarco-test.jsonl
"

eval_cmd="python evaluation/generation/get_rouge.py ${eval_args}"
```

F1 score:
```bash
eval_args="
    -pred ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}/${SAVE_PREFIX}predictions.json
    -gold ${INPUT_DIR}/generation/MSMARCO/${DATA_PREFIX}msmarco-test.jsonl
"

eval_cmd="python evaluation/generation/f1_score.py ${eval_args}"
```

BERTScore:
```bash
eval_args="
  -pred ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}/${SAVE_PREFIX}predictions.json
  -gold ${INPUT_DIR}/generation/MSMARCO/${DATA_PREFIX}msmarco-test.jsonl
  -batch_size 128
  -use_idf
"

eval_cmd="python evaluation/generation/get_bert_score.py ${eval_args}"
```

Entity coverage:
```bash
eval_args="
    -dataset ${INPUT_DIR}/generation/MSMARCO/${DATA_PREFIX}msmarco-test.jsonl
    -pred ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}/${SAVE_PREFIX}predictions.json
    -entity2mention ${INPUT_DIR}/wikipedia/entity2mention.json
"

eval_cmd="python evaluation/generation/get_entity_coverage.py ${eval_args}"
```
The `entity2mention.json` file can be downloaded at [this Google Drive link](https://drive.google.com/file/d/1I8JK_v97soAXd1gXCpcEGDMMRHbxwaAT/view?usp=sharing). To calculate the coverage of unseen entities, add `-remove_input` to the command.
