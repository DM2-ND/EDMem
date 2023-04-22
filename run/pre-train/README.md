# Pre-training

If you wants to pre-train EDMem from scratch on your own data, refer to `src/pretrain.py`, `src/Input.py` (command line arguments) and `src/WikiDatasetBuilder.py` (dataset builder). Besides, we also have scripts to pre-train the memory-based auto-encoder model (EncMem) and the vanilla encoder-decoder model (EncDec) in `src/pretrain_autoencoder.py` and `src/pretrain_nomemory.py` respectively.

Reference command:
```bash
GPUS=$1
CKPT_NAME=$2
MODEL_NAME=$3
LR=$4
SEED=$5

INPUT_DIR=# path to your data
OUTPUT_DIR=# path to your output directory

model_args="
    --model_name_or_path ${MODEL_NAME}
    --entity_embed_size 256
    --elloss_weight 1.0
    --lower_decoder_attn upper_encoder
    --train_from_scratch True
    --inference_nearest_k_entity 100
"

data_args="
    --dataset_dir ${INPUT_DIR}/DIR_OF_YOUR_PROCESS_DATA
    --id2entity_file ${INPUT_DIR}/entid2entityitemid_1M.json
    --already_tokenized True
    --already_processed True
    --process_data_only False
    --save_processed_data False
    --processed_data_dir ${INPUT_DIR}/DIR_OF_YOUR_PROCESS_DATA
    --validation_split_percentage 0.5
    --do_mlm True
    --do_ssm True
    --mlm_probability 0.3
    --ssm_probability 0.5
    --mlm_random_replace 0.0
    --loss_on_mask_tokens False
    --max_seq_length 128
    --pad_to_max_length False
    --add_another_bos True
"

train_args="
    --output_dir ${OUTPUT_DIR}/${CKPT_NAME}/${SEED}
    --overwrite_output_dir False
    --do_train True
    --do_eval True
    --do_predict False
    --evaluation_strategy steps
    --eval_steps 5000
    --per_device_train_batch_size 16
    --per_device_eval_batch_size 16
    --gradient_accumulation_steps 16
    --learning_rate ${LR}
    --weight_decay 0.01
    --max_grad_norm 0.1
    --max_steps 1000000
    --lr_scheduler_type linear
    --warmup_ratio 0.1
    --log_on_each_node False
    --logging_first_step True
    --logging_strategy steps
    --logging_steps 5000
    --logging_nan_inf_filter False
    --save_strategy steps
    --save_steps 10000
    --save_on_each_node False
    --seed ${SEED}
    --fp16 True
    --fp16_full_eval True
    --disable_tqdm True
    --load_best_model_at_end True
    --metric_for_best_model eval_perplexity
    --greater_is_better False
    --prediction_loss_only False
    --predict_with_generate False
    --label_names output_entity_link entity_mention_mask
    --report_to tensorboard
    --ddp_find_unused_parameters False
    --label_smoothing_factor 0.0
"

# Other arguments that worth attention:
# --eval_accumulation_steps
# --dataloader_num_workers
# --resume_from_checkpoint
# --gradient_checkpointing

train_cmd="python -m torch.distributed.launch --nproc_per_node=${GPUS} src/pretrain.py
          ${train_args} ${model_args} ${data_args}"
```
