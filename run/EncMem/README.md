# EncMem

EncMem is the memory-based auto-encoder model used in our experiments as the auto-encoder baseline since the Entity as Experts (EaE) is not publicly available. EncMem has the same MLM, salient span masking and entity linking pre-training objectives, as well as the same hyper-parameters (learning rate, training steps, mask ratio, etc) with EDMem. However, this model still differs from EaE because we used pre-labeled entity mentions while EaE trained a mention detection module as part of it. To train an EncMem model, you can refer to the following command (we use TQA official-dev split as an example):

### Pre-trained Model on Wikipedia

The pre-trained checkpoint can be found here: [Google Drive link](https://drive.google.com/file/d/1qGUFphkBtf4ejPy6rC7KE4Hbc6vKwL2W/view?usp=sharing)

```bash
TASK="tqa"
SEED=3518
DATA_PREFIX="tqa-official-dev/entity_linking/el_"
CKPT_NAME="tqa-official-dev-EncMem_batch256-wait20-warmup0.05-lr1e-5-dropout0.1-eval300"
PRETRAIN_DIR="autoencoder-gpu8-step1M-warmup0.1-batch2048-lr1e-4-norm0.1-ssm0.5-mlm0.3-el1.0/checkpoint-1000000"
INPUT_DIR="data"
# OUTPUT_DIR is the directory of trained checkpoints

if [ "$TASK" = "wq" ] ; then
  BATCH_SIZE=16
  EVAL_STEPS=50
  NUM_EPOCHS=1000
  WAIT_STEPS=30
else
  BATCH_SIZE=32
  EVAL_STEPS=300
  NUM_EPOCHS=300
  WAIT_STEPS=20
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
    --inference_nearest_k_entity 100
    --generate_target entity_linking
    --final_layer_loss answer_only
    --apply_el_loss all
    --prepend_question_in_decoder_input True
    --max_input_length 50
    --max_output_length 20
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

train_cmd="python -m torch.distributed.launch --nproc_per_node=${GPUS} src/run_autoencoder.py ${train_args}"
```
P.S. This is run on 8x V100 GPUs. So the effective batch size is 256. Hyper-parameter search is mainly conducted on `learning_rate`.

### Evaluation
To compare the predicted entity with the ground-truth answer entity in the dataset:

```bash
eval_args="
    -pred ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}/predictions.json
    -gold ${INPUT_DIR}/openqa/${TASK}/${DATA_PREFIX}${TASK}-test.jsonl
"

eval_cmd="python evaluation/openqa/entity_linking_eval.py ${eval_args}"
```

To compare the predicted entity in an OpenQA fashion using exact match to the answer:

```bash
eval_args="
    -pred ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}/predictions.json
    -gold ${INPUT_DIR}/openqa/${TASK}/${DATA_PREFIX}${TASK}-test.jsonl
    -ans_link ${INPUT_DIR}/openqa/${TASK}/tqa-official-dev/entity_linking/sling_invocab-test.json
    -entity_vocab ${INPUT_DIR}/wikipedia/entity_1m.json
"

eval_cmd="python evaluation/openqa/exact_match.py ${eval_args}"
```
