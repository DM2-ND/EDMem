"""
Script to run the model (training / inference).
@Date  : 01/22/2022
@Author: Zhihan Zhang
@mail  : zzhang23@nd.edu
@homepage: ytyz1307zzh.github.io
"""
import gc
import logging
import os
import pdb
import sys
import random
import torch
import pickle
import transformers
import datasets
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from WikiDataset import WikiDataset
from WikiCollator import WikiCollator
from data_utils import load_wiki_data, load_id2entity
from Model import EMAGModel
from Input import ModelArguments, DataArguments
from Output import EMAGModelOutput
from Seq2SeqTrainer import Seq2SeqTrainer
from training_utils import compute_recall_metrics, combine_recall_metrics

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set the maximum memory size that `datasets` library can use to load data files
    datasets.config.IN_MEMORY_MAX_SIZE = data_args.datasets_inmemory_maxsize
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = str(data_args.datasets_inmemory_maxsize)

    model_args.fp16 = training_args.fp16
    model_args.generate_target = "pretraining"  # to align with QA arguments

    if data_args.enable_mpi:
        # Enable MPI for multi-node GPU training
        from MPIAdapter import MPIAdapter
        adapter = MPIAdapter()
        training_args.local_rank = adapter.local_rank

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get the tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Add entity-related special tokens <E_s> and <E_e> to the tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": ["<E_s>", "<E_e>"]})
    entity_start_token_id = tokenizer.convert_tokens_to_ids('<E_s>')
    entity_end_token_id = tokenizer.convert_tokens_to_ids('<E_e>')
    # Add these properties to data_args for the dataset object to use
    data_args.entity_start_token_id = entity_start_token_id
    data_args.entity_end_token_id = entity_end_token_id
    data_args.entity_start_token = '<E_s>'
    data_args.entity_end_token = '<E_e>'
    # Also add to model_args for EntityMemory to use
    model_args.entity_start_token_id = entity_start_token_id
    model_args.entity_end_token_id = entity_end_token_id

    # Get the id2entity dictionary
    id2entity = load_id2entity(data_args.id2entity_file)

    # Get the dataset
    # if training and validation file paths are specified:
    # if data_args.train_file is not None and data_args.validation_file is not None:
    #     train_datalist = load_wiki_data(data_args.train_file)
    #     random.shuffle(train_datalist)
    #     eval_datalist = load_wiki_data(data_args.validation_file)
    # elif data_args.dataset_dir is not None:
    #     assert data_args.train_file is None and data_args.validation_file is None
    #     whole_datalist = load_wiki_data(data_args.dataset_dir)
    #     logger.info(f"Read a total of {len(whole_datalist)} data.")
    #     random.shuffle(whole_datalist)
    #     eval_data_size = round((data_args.validation_split_percentage / 100) * len(whole_datalist))
    #     eval_datalist = whole_datalist[:eval_data_size]
    #     train_datalist = whole_datalist[eval_data_size:]
    #     logger.info(f"Training set size: {len(train_datalist)}, evaluation set size: {len(eval_datalist)}")
    #     del whole_datalist
    # else:
    #     raise ValueError("Invalid values for dataset_dir, train_file and validation_file")
    #
    # # If max_train_samples/max_eval_samples is set, select a part of the original dataset (mainly for debugging)
    # if data_args.max_train_samples is not None:
    #     train_datalist = train_datalist[:data_args.max_train_samples]
    # if data_args.max_eval_samples is not None:
    #     eval_datalist = eval_datalist[:data_args.max_eval_samples]
    #
    # # Build WikiDataset objects
    # if training_args.do_train:
    #     train_dataset = WikiDataset(args=data_args,
    #                                 dataset=train_datalist,
    #                                 id2entity=id2entity,
    #                                 logger=logger,
    #                                 tokenizer=tokenizer,
    #                                 already_tokenized=data_args.already_tokenized,
    #                                 is_training=True)
    #
    # if training_args.do_eval:
    #     eval_dataset = WikiDataset(args=data_args,
    #                                dataset=eval_datalist,
    #                                id2entity=id2entity,
    #                                logger=logger,
    #                                tokenizer=tokenizer,
    #                                already_tokenized=data_args.already_tokenized,
    #                                is_training=False)

    added_special_tokens = {"entity_start_token": "<E_s>", "entity_end_token": "<E_e>"}

    # If the dataset is already processed (scaned by `load_dataset` function) and saved to disk using
    # `save_to_disk` function, then we can directly load it using `load_from_disk` which is faster
    if data_args.already_processed:
        assert data_args.dataset_dir is not None
        raw_datasets = load_from_disk(data_args.dataset_dir)
    else:
        if data_args.process_data_only:
            dummy_tensor = torch.rand(30000, 100000)
            dummy_tensor.to(training_args.device)

        # Scan and read the dataset using `load_dataset`
        if data_args.dataset_dir is not None:
            raw_datasets = load_dataset(
                "src/WikiDatasetBuilder.py",
                name="default",
                data_dir=data_args.dataset_dir,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                special_tokens=added_special_tokens,
                already_tokenized=data_args.already_tokenized,
                validation_split_percentage=data_args.validation_split_percentage,
                add_another_bos=data_args.add_another_bos
            )
        else:
            raw_datasets = load_dataset(
                "src/WikiDatasetBuilder.py",
                name="default",
                data_files={"train": data_args.train_file, "validation": data_args.validation_file},
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                special_tokens=added_special_tokens,
                already_tokenized=data_args.already_tokenized,
                validation_split_percentage=data_args.validation_split_percentage,
                add_another_bos=data_args.add_another_bos
            )
        # If specified, save the dataset to disk for fast loading next time (default: True)
        if data_args.save_processed_data and training_args.local_rank == 0:
            assert data_args.processed_data_dir is not None
            if not os.path.exists(data_args.processed_data_dir):
                os.makedirs(data_args.processed_data_dir)
            raw_datasets.save_to_disk(data_args.processed_data_dir)
        # If specified, exit the program after we process the data
        if data_args.process_data_only:
            logger.info("Finished processing data. Exit...")
            return

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        logger.info(f"Training set size: {len(train_dataset)}")

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        logger.info(f"Validation set size: {len(eval_dataset)}")

    # Get the configuration object for transformer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # Load pre-trained entity embedding
    entity_embedding = None
    if model_args.entity_embedding_file is not None:
        entity_embedding = pickle.load(open(model_args.entity_embedding_file, 'rb'))

    # Load the model
    model = EMAGModel(config=config, model_args=model_args, entity_embedding=entity_embedding)
    # Resize the embedding table of the model
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    del entity_embedding

    # Set the max sequence length for truncation/padding.
    # NOTE: Only pad to this value if data_args.pad_to_max_length is set
    if data_args.max_seq_length is None:
        if data_args.pad_to_max_length:
            raise ValueError("pad_to_max_length is True but no max_pad_length is specified.")
        logger.warning(
            f"You did not specify max_seq_length. Will not truncate the input sequence."
        )
        max_seq_length = None
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        if not data_args.pad_to_max_length:
            logger.warning(
                f"Specified max_seq_length but pad_to_max_length is False. Will only apply max_seq_length to "
                f"truncation but not to padding. Dynamic padding will be used."
            )

    # Data collator: padding and masking
    pad_to_multiple_of_8 = training_args.fp16 and not data_args.pad_to_max_length
    if pad_to_multiple_of_8:
        logger.warning("training_args.fp16=True and data_args.pad_to_max_length=False: "
                       "input sequences will be padded to multiples of 8.")
    elif data_args.pad_to_max_length:
        logger.warning(f"Input sequences will be padded to max_seq_length {max_seq_length}.")
    else:
        logger.warning(f"Dynamic padding: input sequences will be padded to the longest length in a batch.")
    data_collator = WikiCollator(
        tokenizer=tokenizer,
        mlm=data_args.do_mlm,
        ssm=data_args.do_ssm,
        mlm_probability=data_args.mlm_probability,
        ssm_probability=data_args.ssm_probability,
        mlm_random_replace=data_args.mlm_random_replace,
        loss_on_mask_tokens=data_args.loss_on_mask_tokens,
        max_seq_length=max_seq_length if data_args.pad_to_max_length else None,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None
    )

    def compute_metrics(eval_preds):
        prediction_tuple = eval_preds.predictions
        encoder_topk_entity, decoder_topk_entity, final_topk_entity = prediction_tuple
        labels, mention_mask = eval_preds.label_ids

        # Get only the predictions and labels on the masked out entities
        assert labels.shape[0] == encoder_topk_entity.shape[0] == decoder_topk_entity.shape[0] \
               == final_topk_entity.shape[0] == mention_mask.shape[0]
        labels = labels[mention_mask]
        encoder_topk_entity = encoder_topk_entity[mention_mask]
        decoder_topk_entity = decoder_topk_entity[mention_mask]
        final_topk_entity = final_topk_entity[mention_mask]
        num_mentions = labels.shape[0]

        encoder_recall = compute_recall_metrics(encoder_topk_entity, labels)
        decoder_recall = compute_recall_metrics(decoder_topk_entity, labels)
        final_recall = compute_recall_metrics(final_topk_entity, labels)

        recall_metrics = combine_recall_metrics(encoder_recall, decoder_recall, final_recall)
        recall_metrics["num_mentions"] = torch.tensor(num_mentions).to(labels.device)

        return recall_metrics

    # the trainer. compute_metrics is set to None to use forward()
    # instead of generate() during evaluation. Therefore, evaluation loss
    # will be returned for calculating perplexity
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # Output fields to ignore during evaluation
    ignore_keys_for_eval = [key for key in EMAGModelOutput.key_names() if key not in
                            ["loss", "detail_loss", "topk_entity"]]

    # do train
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(logger=logger,
                                     resume_from_checkpoint=checkpoint,
                                     ignore_keys_for_eval=ignore_keys_for_eval)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # do eval (if load_best_model_at_end = True, then here is the evaluation of the best checkpoint)
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # If we directly do evaluation without training, we need to load a checkpoint first
        if not training_args.do_train:
            load_model_first = True
            checkpoint = data_args.checkpoint_dir
        else:
            load_model_first = False
            checkpoint = None

        metrics = trainer.evaluate(logger=logger,
                                   ignore_keys=ignore_keys_for_eval,
                                   checkpoint=checkpoint,
                                   load_model_first=load_model_first)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
