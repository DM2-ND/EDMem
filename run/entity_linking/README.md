# Entity Linking

EDMem can also be used as an entity linking model. It can select an entity from the entity memory via the entity linking head on the top of the decoder. This is actually the first step of the two-step decoding in *static entity linking*. To train an entity linking model, you can refer to the following command (we use TQA official-dev split as an example):

```bash
TASK="tqa"
SEED=3518
DATA_PREFIX="tqa-official-dev/entity_linking/el_"
CKPT_NAME="tqa-official-dev-EL_batch256-wait30-warmup0.05-lr1e-5-dropout0.1-eval300"
PRETRAIN_DIR="scratch-attn_upper-gpu8-step1m-warmup0.1-batch2048-lr1e-4-norm0.1-ssm0.5-mlm0.3-el1.0/checkpoint-1000000"
INPUT_DIR="data"
# OUTPUT_DIR is the directory of trained checkpoints

if [ "$TASK" = "wq" ] ; then
  BATCH_SIZE=16
  EVAL_STEPS=50
  NUM_EPOCHS=1000
  WAIT_STEPS=30
else
  EVAL_STEPS=300
  NUM_EPOCHS=300
  WAIT_STEPS=30
  BATCH_SIZE=32
fi

train_args="
    --data_dir ${INPUT_DIR}/openqa
    --id2entity_file ${INPUT_DIR}/wikipedia/entid2entityitemid_1M.json
    --id2entitytokens_file ${INPUT_DIR}/wikipedia/entity_tokens_1m.json
    --task ${TASK}
    --datafile_prefix ${DATA_PREFIX}
    --do_train
    --do_predict
    --output_dir ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}
    --model_name facebook/bart-large
    --pretrain_ckpt ${OUTPUT_DIR}/${PRETRAIN_DIR}
    --entity_embed_size 256
    --dropout 0.1
    --elloss_weight 1.0
    --entity_token_weight 1.0
    --lower_decoder_attn upper_encoder
    --inference_nearest_k_entity 100
    --generate_target entity_linking
    --final_layer_loss answer_only
    --apply_el_loss all
    --prepend_question_in_decoder_input True
    --max_input_length 50
    --max_output_length 20
    --num_beams 5
    --do_sample False
    --add_another_bos True
    --train_batch_size ${BATCH_SIZE}
    --predict_batch_size 32
    --learning_rate 1e-5
    --weight_decay 0.01
    --max_grad_norm 0.1
    --gradient_accumulation_steps 1
    --num_train_epochs ${NUM_EPOCHS}
    --warmup_ratio 0.05
    --wait_step ${WAIT_STEPS}
    --eval_period ${EVAL_STEPS}
    --seed ${SEED}
"

train_cmd="python -m torch.distributed.launch --nproc_per_node=${GPUS} src/run_QA.py ${train_args}"
```
P.S. This is run on 8x V100 GPUs. So the effective batch size is 256. Hyper-parameter search is mainly conducted on `learning_rate`.

### Evaluation

```bash
eval_args="
    -pred ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}/predictions.json
    -gold ${INPUT_DIR}/openqa/${TASK}/${DATA_PREFIX}${TASK}-test.jsonl
    -ans_link ${INPUT_DIR}/openqa/${TASK}/tqa-official-dev/entity_linking/sling_invocab-test.json
    -entity_vocab ${INPUT_DIR}/wikipedia/entity_1m.json
"

eval_cmd="python evaluation/openqa/exact_match.py ${eval_args}"
```
