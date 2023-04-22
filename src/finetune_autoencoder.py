"""
Training the autoencoder model & predicting on OpenQA datasets.
@Date  : 06/09/2022
@Author: Zhihan Zhang
@mail  : zzhang23@nd.edu
@homepage: ytyz1307zzh.github.io
"""

import os
import re
import pdb
import time
import math
import numpy as np
import torch
import pickle
from tqdm import tqdm

from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup

from QA_utils import (
    load_dataset,
    load_all_datasets,
    mean_reduce,
    sum_reduce,
    combine_prediction_files,
    load_id2entity,
    load_id2entitytokens,
    calculate_num_steps,
    prepare_inputs,
    convert_autoencoder_inputs,
    mean
)
from model_utils import set_autoencoder_config
from QADataset import QADataset, QADataLoader
from QACollator import QACollator
from Loss import DetailLoss
from Autoencoder import EMAGAutoEncoder


def run(args, logger):

    # initialize DDP
    if args.n_gpu > 1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    config = BartConfig.from_pretrained(args.model_name)
    config = set_autoencoder_config(config)
    config.dropout = args.dropout  # Set the dropout rate to what we input from command line
    tokenizer = BartTokenizer.from_pretrained(args.model_name)

    tokenizer.add_special_tokens({"additional_special_tokens": ["<E_s>", "<E_e>"]})
    entity_start_token_id = tokenizer.convert_tokens_to_ids('<E_s>')
    entity_end_token_id = tokenizer.convert_tokens_to_ids('<E_e>')
    # Add these properties to data_args for the dataset object to use
    args.entity_start_token_id = entity_start_token_id
    args.entity_end_token_id = entity_end_token_id
    args.entity_start_token = '<E_s>'
    args.entity_end_token = '<E_e>'

    # Add these arguments to align with the pre-training phase. Otherwise AttributeError will be raised
    args.elloss_weight = 1.0
    args.entity_token_weight = None

    # Get the id2entity dictionary
    id2entity = load_id2entity(args.id2entity_file)
    id2entitytokens = load_id2entitytokens(args.id2entitytokens_file)
    entity_mask = torch.load(args.entity_mask_file) if args.entity_mask_file is not None else None

    if args.task == "all":
        raw_datasets = load_all_datasets(data_dir=args.data_dir,
                                         datafile_prefix=args.datafile_prefix)
        train_datalist = raw_datasets["train"]
        dev_nq_datalist = raw_datasets["nq_dev"]
        dev_tqa_datalist = raw_datasets["tqa_dev"]
        dev_wq_datalist = raw_datasets["wq_dev"]
        test_nq_datalist = raw_datasets["nq_test"]
        test_tqa_datalist = raw_datasets["tqa_test"]
        test_wq_datalist = raw_datasets["wq_test"]

        train_data = QADataset(args, train_datalist, id2entity, logger, tokenizer, entity_mask, is_training=True)
        dev_nq_data = QADataset(args, dev_nq_datalist, id2entity, logger, tokenizer, entity_mask, is_training=False)
        dev_tqa_data = QADataset(args, dev_tqa_datalist, id2entity, logger, tokenizer, entity_mask, is_training=False)
        dev_wq_data = QADataset(args, dev_wq_datalist, id2entity, logger, tokenizer, entity_mask, is_training=False)
        test_nq_data = QADataset(args, test_nq_datalist, id2entity, logger, tokenizer, entity_mask, is_training=False)
        test_tqa_data = QADataset(args, test_tqa_datalist, id2entity, logger, tokenizer, entity_mask, is_training=False)
        test_wq_data = QADataset(args, test_wq_datalist, id2entity, logger, tokenizer, entity_mask, is_training=False)
        dev_data = [dev_nq_data, dev_tqa_data, dev_wq_data]
        test_data = [test_nq_data, test_tqa_data, test_wq_data]

    else:
        raw_datasets = load_dataset(data_dir=args.data_dir,
                                    task_name=args.task,
                                    datafile_prefix=args.datafile_prefix)
        train_datalist = raw_datasets["train"]
        dev_datalist = raw_datasets["dev"]
        test_datalist = raw_datasets["test"]

        if args.debug:
            train_datalist = train_datalist[:10]
            dev_datalist = train_datalist[:10]
            test_datalist = train_datalist[:10]

        train_data = QADataset(args, train_datalist, id2entity, logger, tokenizer, entity_mask, is_training=True)
        dev_data = QADataset(args, dev_datalist, id2entity, logger, tokenizer, entity_mask, is_training=False)
        test_data = QADataset(args, test_datalist, id2entity, logger, tokenizer, entity_mask, is_training=False)

    data_collator = QACollator(config=config, generate_target=args.generate_target)

    if args.do_train:
        # if continue training, load the previous checkpoint from args.output_dir
        # if args.output_dir is not None \
        #     and args.continue_training \
        #         and os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")):
        #     model = EMAGModel.from_pretrained(args.output_dir)
        # Otherwise, load the pre-trained model and start fine-tuning
        # else:
        model = EMAGAutoEncoder(config=config, model_args=args)
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        if args.n_gpu > 1:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        elif args.n_gpu == 1:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
        else:
            map_location = {'cuda:%d' % 0: 'cpu'}
        state_dict = torch.load(os.path.join(args.pretrain_ckpt, "pytorch_model.bin"), map_location=map_location)
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from {}".format(args.pretrain_ckpt))

        if torch.cuda.is_available():
            # for multi-gpu single-node training, use DDP
            if args.n_gpu > 1:
                model.to(device)
                logger.warning(f"Moving model to {device}...")
                model = torch.nn.parallel.DistributedDataParallel(model,
                                                                  device_ids=[args.local_rank],
                                                                  output_device=args.local_rank,
                                                                  find_unused_parameters=False)
            # otherwise, only use one gpu
            else:
                model.to(torch.device("cuda"))

        no_decay = ['bias', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        train_steps = calculate_num_steps(args, len(train_data))
        args.total_global_steps = train_steps["total_global_steps"]
        args.total_local_steps = train_steps["total_local_steps"]
        args.warmup_steps = train_steps["warmup_steps"]
        logger.info(f"Total global training steps: {args.total_global_steps}, "
                    f"Total local training steps: {args.total_local_steps}, "
                    f"Warmup steps: {args.warmup_steps}")
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.total_global_steps)
        train(args, logger, model, train_data, dev_data, data_collator,
              optimizer, scheduler, tokenizer, id2entity, entity_mask)

    # sync all processes if continue to do prediction
    if args.do_train and args.do_predict and args.n_gpu > 1:
        torch.distributed.barrier()

    if args.do_predict:
        model = EMAGAutoEncoder(config=config, model_args=args)
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        if args.n_gpu > 1:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        elif args.n_gpu == 1:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
        else:
            map_location = {'cuda:%d' % 0: 'cpu'}
        state_dict = torch.load(os.path.join(args.output_dir, "pytorch_model.bin"), map_location=map_location)
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from {}".format(args.output_dir))

        if torch.cuda.is_available():
            # for multi-gpu single-node training, use DDP
            if args.n_gpu > 1:
                model.to(device)
                model = torch.nn.parallel.DistributedDataParallel(model,
                                                                  device_ids=[args.local_rank],
                                                                  output_device=args.local_rank,
                                                                  find_unused_parameters=False)
            # otherwise, only use one gpu
            else:
                model.to(torch.device("cuda"))

        model.eval()

        # If perform prediction on dev set
        if args.predict_on_dev:
            test_data = dev_data

        # If we do multi-task training, then calcualte the average EM on three datasets
        if args.task == "all":
            all_ems = []
            for data in test_data:
                single_em = entity_linking_inference(
                    args,
                    model if args.n_gpu <= 1 else model.module,
                    data,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    id2entity=id2entity,
                    entity_mask=entity_mask,
                    logger=logger,
                    is_test=True
                )
                all_ems.append(single_em)
            logger.info(f"NQ EM: {all_ems[0]*100:.2f}, TQA EM: {all_ems[1]*100:.2f}, WQ EM: {all_ems[2]*100:.2f}")
            ems = mean(all_ems)

        else:
            ems = entity_linking_inference(
                args,
                model if args.n_gpu <= 1 else model.module,
                test_data,
                data_collator=data_collator,
                tokenizer=tokenizer,
                id2entity=id2entity,
                entity_mask=entity_mask,
                logger=logger,
                is_test=True
            )
        if args.predict_on_dev:
            logger.info("EM on dev data: %.2f%%" % (ems * 100))
        else:
            logger.info("EM on test data: %.2f%%" % (ems*100))

        # sync all processes before combining prediction files
        if args.n_gpu > 1:
            torch.distributed.barrier()

        if args.n_gpu > 1 and args.local_rank == 0:
            if args.task == "all":
                for dataset_name in ["nq", "tqa", "wq"]:
                    combine_prediction_files(args.output_dir, args.n_gpu, args.prefix + dataset_name)
            else:
                combine_prediction_files(args.output_dir, args.n_gpu, args.prefix)


def train(args, logger, model, train_data, dev_data, data_collator, optimizer, scheduler, tokenizer, id2entity, entity_mask):
    model.train()
    global_step = 0
    local_step = 0
    last_logged_step = 0
    train_loss_host = torch.tensor(0.0).to(model.device)
    detail_loss_host = DetailLoss().to(model.device)
    best_accuracy = -1
    stop_training = False

    train_dataloader = QADataLoader(args, train_data, collate_fn=data_collator, is_training=True)
    train_start_time = time.time()

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        logger.info('*'*20 + f"Epoch {epoch+1}" + "*"*20)
        for batch in train_dataloader:
            local_step += 1
            batch["decoder_input_ids"], batch["output_entity_link"] = convert_autoencoder_inputs(
                batch["decoder_input_ids"], batch["output_entity_link"], tokenizer=tokenizer
            )
            batch = prepare_inputs(batch, model.device)
            if entity_mask is not None:
                entity_mask = entity_mask.to(model.device)
            # print(f"GPU{model.device} Batch: {batch['id']}")
            # the input of the autoencoder model is the same with the decoder-side input of the encoder-decoder model
            # when generate_target == "entity_linking"
            model_output = model(input_ids=batch["decoder_input_ids"],
                                 attention_mask=batch["decoder_attention_mask"],
                                 input_entity_link=batch["output_entity_link"],
                                 entity_mask=entity_mask,
                                 return_dict=True)
            loss = model_output.loss
            detail_loss = model_output.detail_loss

            # print(f"GPU{model.device} Loss: {loss}")
            # print(f"GPU{model.device} Detail Loss: {detail_loss}")

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                detail_loss = detail_loss / args.gradient_accumulation_steps
                # print(f"GPU{model.device} Loss for Gradient Accu: {loss}")
                # print(f"GPU{model.device} Detail Loss for Gradient Accu: {detail_loss}")

            # get the average loss value across gpus (just for logging),
            # the distributed backward pass is handled by DDP itself
            loss_value = loss.clone().detach()
            detail_loss = detail_loss.detach()

            mean_reduce(loss_value, args)
            detail_loss = DetailLoss([mean_reduce(x, args) for x in detail_loss.to_tupled_tensor()])

            # print(f"GPU{model.device} Mean Loss: {loss_value}")
            # print(f"GPU{model.device} Mean Detail Loss: {detail_loss}")

            # if args.n_gpu > 1:
            #     assert loss.dim() == 1  # loss is a 1-dim tensor
            #     loss = loss.mean()  # mean() to average on multi-gpu.

            if torch.isnan(loss).data:
                logger.error("Stop training because loss=%s" % loss.data)
                stop_training = True
                break

            # store the history loss values on main process (for logging purpose)
            train_loss_host += loss_value
            detail_loss_host += detail_loss
            # print(f"GPU{model.device} Total Loss: {train_loss_host}")
            # print(f"GPU{model.device} Total Detail Loss: {detail_loss_host}")
            # print("Train_losses: ", train_losses)
            loss.backward()

            # gradient accumulation
            # Always make an optimization step at the last batch of training
            if local_step % args.gradient_accumulation_steps == 0 or \
                    local_step == args.total_local_steps:
                # print(f"GPU{model.device} Doing optimization. Global step: {global_step}, local step: {local_step}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()
                global_step += 1

                # evaluate on dev set
                # Always evaluate at the last batch of training
                if global_step % args.eval_period == 0 or \
                        global_step == args.total_global_steps:
                    # print(f"GPU{model.device} Doing evaluation. Global step: {global_step}, local step: {local_step}")

                    model.eval()
                    train_duration = time.time() - train_start_time
                    logger.info(f"Train duration: {train_duration:.3f}s")

                    # If we do multi-task training, then calcualte the average EM on three datasets
                    if args.task == "all":
                        all_ems = []
                        for data in dev_data:
                            single_em = entity_linking_inference(
                                args,
                                model if args.n_gpu <= 1 else model.module,
                                data,
                                data_collator=data_collator,
                                logger=logger,
                                tokenizer=tokenizer,
                                id2entity=id2entity,
                                entity_mask=entity_mask
                            )
                            all_ems.append(single_em)
                        logger.info(f"NQ EM: {all_ems[0]*100:.2f}, TQA EM: {all_ems[1]*100:.2f}, "
                                    f"WQ EM: {all_ems[2]*100:.2f}")
                        curr_em = mean(all_ems)

                    else:
                        curr_em = entity_linking_inference(
                            args,
                            model if args.n_gpu <= 1 else model.module,
                            dev_data,
                            data_collator=data_collator,
                            logger=logger,
                            tokenizer=tokenizer,
                            id2entity=id2entity,
                            entity_mask=entity_mask
                        )
                    # Logging
                    logger.info("Step %d, Train loss %.2f, EM %.2f%%, Detail loss: %s, epoch=%d" % (
                            global_step,
                            train_loss_host / (global_step - last_logged_step),
                            curr_em * 100,
                            detail_loss_host / (global_step - last_logged_step),
                            epoch + 1))

                    # Store the current step as last_logged_step, clear the loss hosts
                    last_logged_step = global_step
                    train_loss_host -= train_loss_host
                    detail_loss_host.clear()

                    if best_accuracy < curr_em:
                        # while DDP, only save model using the main process
                        if args.n_gpu > 1 and args.local_rank == 0:
                            model.module.save_pretrained(args.output_dir)
                        elif args.n_gpu <= 1:
                            model.save_pretrained(args.output_dir)
                        # Logging
                        logger.info("Saving model with best EM: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" %
                                    (best_accuracy*100.0, curr_em*100.0, epoch + 1, global_step))
                        logger.info("Checkpoint saved to {}".format(args.output_dir))

                        best_accuracy = curr_em
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        # early stopping if accuracy did not improve for args.wait_step evaluation rounds
                        if wait_step >= args.wait_step:
                            stop_training = True
                            break
                    model.train()
                    train_start_time = time.time()
        if stop_training:
            break
    logger.info("Best EM on validation data: %.2f" % (best_accuracy * 100))


def entity_linking_inference(args, model, dev_data, data_collator, tokenizer, id2entity,
                             entity_mask, logger, is_test=False):
    predictions = []
    raw_predictions = []
    data_ids = []
    data_cnt = 0  # total number of instances in this node

    dev_dataloader = QADataLoader(args, dev_data, collate_fn=data_collator, is_training=False)
    eval_start_time = time.time()

    assert args.generate_target == "entity_linking"

    for i, batch in enumerate(dev_dataloader):
        batch["decoder_input_ids"] = convert_autoencoder_inputs(batch["decoder_input_ids"], tokenizer=tokenizer)
        batch = prepare_inputs(batch, model.device)
        if entity_mask is not None:
            entity_mask = entity_mask.to(model.device)
        # Items in inference batch: id, input_ids, attention_mask, input_entity_link
        model_output = model(input_ids=batch["decoder_input_ids"],
                             attention_mask=batch["decoder_attention_mask"],
                             entity_mask=entity_mask,
                             return_dict=True)

        # predicted top-k entities for the answer, shape (batch, topk)
        topk_entity = model_output.topk_entity[1]
        predicted_entity = topk_entity[:, 0].tolist()  # pick top-1

        # print(f"GPU{args.local_rank} Output: ", outputs)
        # decode token ids to strings
        for data_id, predicted_eid, topk_pred in zip(batch["id"], predicted_entity, topk_entity):
            entity_name = id2entity[predicted_eid]
            predictions.append(entity_name)
            raw_predictions.append([entity_name, topk_pred.tolist()])
            data_ids.append(data_id)
        data_cnt += batch["decoder_input_ids"].size(0)  # add the size of the current batch
        # print(f'GPU{args.local_rank} predictions: {predictions}')
        # print(f'GPU{args.local_rank} data_ids: {data_ids}')
        # print(f'GPU{args.local_rank} data_cnt: {data_cnt}')

    # if multi-task training, we need to figure out which dataset we are saving
    if args.task == "all":
        dataset_name = batch["id"][0].split('_')[0]
        save_prefix = args.prefix + dataset_name
    else:
        save_prefix = args.prefix

    # if multi-gpu, only do the saving on main process
    if is_test:
        if args.n_gpu <= 1:
            save_path = os.path.join(args.output_dir, "{}predictions.json".format(save_prefix))
            dev_data.save_predictions(data_ids, raw_predictions, save_path)
        if args.n_gpu > 1:
            save_path = os.path.join(args.output_dir, "{}predictions_ps{}.json".format(save_prefix, args.local_rank))
            dev_data.save_predictions(data_ids, raw_predictions, save_path)
    # calculate EM score, use data_id to find ground-truth answer and compare with predictions
    score = np.mean(dev_data.evaluate(data_ids, predictions))

    eval_duration = time.time() - eval_start_time
    logger.info(f"Eval duration: {eval_duration:.3f}s")

    # if multi-gpu, we need to compute weighted average EM across gpus
    if args.n_gpu > 1:
        score = torch.tensor(score).to(model.device)
        score = score * data_cnt
        # weighted sum + divide by total number of data objects for average EM
        sum_reduce(score, args)
        data_size = math.ceil(len(dev_data) / args.n_gpu) * args.n_gpu
        score = score / data_size
        # send the final EM to all processes
        # torch.distributed.broadcast(score, src=0)
        return score.item()
    else:
        return score
