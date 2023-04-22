"""
Script for running the autoencoder model.
@Date  : 06/09/2022
@Author: Zhihan Zhang
@mail  : zzhang23@nd.edu
@homepage: ytyz1307zzh.github.io
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random
import numpy as np
import torch
import transformers

from finetune_autoencoder import run
from Constants import ENTITY_VOCAB_SIZE
torch.set_printoptions(threshold=1000, edgeitems=6)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v in ["true", "True"]:
        return True
    elif v in ["false", "False"]:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v} instead.")


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    # parser.add_argument("--train_file", type=str, required=True)
    # parser.add_argument("--dev_file", type=str, required=True)
    # parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--id2entity_file", default=None, help="Path to the id2entity file.")
    parser.add_argument("--id2entitytokens_file", default=None, help="Path to the id2entitytokens file.")
    parser.add_argument("--task", choices=["nq", "tqa", "wq", "all"], required=True)
    parser.add_argument("--datafile_prefix", default="",
                        help="Specify this if there is a prefix before data files, i.e., data files are not named "
                             "as `nq-train.jsonl` but `prefix_nq-train.jsonl` (take NQ as an example).")
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--overwrite_output", default=False, type=str2bool,
                        help="If `True`, will overwrite the previous checkpoints (if exist).")
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--predict_on_dev", action='store_true', default=False,
                        help='Specify to make predictions on dev set.')
    # parser.add_argument("--continue_training", type=str2bool, default=False,
    #                     help="Specify it to continue training from a previous checkpoint.")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="facebook/bart-large",
                        help="Huggingface model name")
    parser.add_argument("--pretrain_ckpt", type=str, required=True, help="Directory of pre-trained EMAG model")
    parser.add_argument("--num_lower_layers", type=int, default=4,
                        help="The number of transformer layers in the lower block.")
    parser.add_argument("--num_upper_layers", type=int, default=8,
                        help="The number of transformer layers in the upper block.")
    parser.add_argument("--entity_embed_size", type=int, default=256,
                        help="The dimension of entity embeddings in the entity memory.")
    parser.add_argument("--entity_vocab_size", type=int, default=ENTITY_VOCAB_SIZE,
                        help="The size of entity memory.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--entity_mask_file", default=None,
                        help="Path to a file containing a mask vector on the entity vocabulary")
    parser.add_argument("--inference_nearest_k_entity", type=int, default=100,
                        help="During inference, the number of candidate entities while accessing memory."
                             "We do not calcualte attention on all 1M entities during inference.")
    parser.add_argument("--do_cased", type=str2bool, default=False,
                        help="If `True`, inputs and outputs will not be lowercased.")

    # Preprocessing/decoding-related parameters
    parser.add_argument('--generate_target', choices=["entity_linking"], default="entity_linking",
                        help="The target output. entity_linking: predict an entity id. "
                             "For fine-tuning the autoencoder, model, entity_linking is the only choice. "
                             "Keeping this argument is to distinguish fine-tuning from pre-training.")
    parser.add_argument('--max_input_length', type=int, default=50)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument("--add_another_bos", default=True, type=str2bool,
                        help="If `True`, will add another BOS token at the beginning.")

    # Entity linking task
    parser.add_argument("--final_layer_loss", choices=["answer_only", "all"], default="answer_only",
                        help="Whether to apply loss on all entity mentions (including those in the question) "
                             "in the final-layer entity linking loss. This argument only works when "
                             "--generate_target='entity_linking'.")
    parser.add_argument("--apply_el_loss", choices=["final", "all"], default="all",
                        help="Whether to compute entity linking loss on all layers. "
                             "This argument only works when --generate_target='entity_linking'.")
    parser.add_argument("--prepend_question_in_decoder_input", type=str2bool, default=True,
                        help="Whether to prepend the question to the decoder input "
                             "(before the <E_s> token used for entity prediction)")

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="The number of steps used for warmup.")
    parser.add_argument('--wait_step', type=int, default=10,
                        help="If evaluation metric does not improve for this number of global steps, "
                             "stop the training process.")

    # Other parameters
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    # Logging levels: DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
    parser.add_argument('--log_level', default=logging.INFO, type=int, help="Log level on main node.")
    parser.add_argument('--log_level_replica', default=logging.WARNING, type=int, help="Log level on replica node.")
    parser.add_argument('--eval_period', type=int, default=1000,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    # whether to lowercase the dataset
    args.do_lowercase = not args.do_cased

    # These arguments are used by EMAG model in pre-training, we set them to
    # some "dull" values so that it won't affect the fine-tuning stage
    args.fp16 = False  # This script currently does not support fp16
    args.train_from_scratch = True  # This has no effect since we will load the pre-trained model

    # Create output_dir for logging and saving checkpoints
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Start writing logs
    log_filename = "{}log.txt".format("" if args.do_train else args.prefix)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    if args.local_rank == -1 or args.local_rank == 0:
        log_level = args.log_level
    else:
        log_level = args.log_level_replica
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.n_gpu = torch.cuda.device_count()

    if args.do_train:
        if os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")):
            logger.warning(f"Output directory {args.output_dir} already exists and has previous checkpoints!")
            if not args.overwrite_output:
                raise FileExistsError(f"Output directory {args.output_dir} already exists and has previous checkpoints!")
            else:
                logger.warning("Previous checkpoints will be overwritten!")

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1 or args.local_rank == 0:
        with open(os.path.join(args.output_dir, "args"), 'w', encoding='utf8') as fargs:
            print(args, file=fargs)

    logger.info("Using {} gpus".format(args.n_gpu))
    run(args, logger)


if __name__ == '__main__':
    main()
