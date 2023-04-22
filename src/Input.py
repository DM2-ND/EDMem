from dataclasses import dataclass, field, asdict
from typing import Optional
from Constants import ENTITY_VOCAB_SIZE


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # model settings
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    entity_embedding_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Entity embeddings obtained from a pre-trained model"
        },
    )

    # EMAG specific settings
    num_lower_layers: int = field(
        default=4,
        metadata={"help": "The number of transformer layers in the lower block."},
    )
    num_upper_layers: int = field(
        default=8,
        metadata={"help": "The number of transformer layers in the upper block."},
    )
    entity_embed_size: int = field(
        default=256,
        metadata={"help": "The dimension of entity embeddings in the entity memory."},
    )
    entity_vocab_size: int = field(
        default=ENTITY_VOCAB_SIZE,
        metadata={"help": "The size of entity memory."},
    )
    elloss_weight: float = field(
        default=1.0,
        metadata={"help": "The coefficient of entity linking loss"},
    )
    entity_token_weight: Optional[float] = field(
        default=None,
        metadata={"help": "The weight for <E_s> token in LM loss"},
    )
    lower_decoder_attn: str = field(
        default="lower_encoder",
        metadata={
            "help": "Which encoder output (lower encoder / upper encoder) should the lower decoder attend to",
            "choices": ["lower_encoder", "upper_encoder"]
        },
    )
    inference_nearest_k_entity: int = field(
        default=100,
        metadata={"help": "During inference, the number of candidate entities while accessing memory."
                          "We do not calcualte attention on all 1M entities during inference."}
    )

    # Arguments that are not used in pre-training phase
    generate_target: Optional[str] = field(
        default=None,
        metadata={"help": "This argument is not used in pre-training. Just add it to be consistent with fine-tuning."}
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=3,
        metadata={"help": "This argument is not used in pre-training. Just add it to be consistent with fine-tuning."}
    )

    # other settings
    train_from_scratch: Optional[bool] = field(
        default=False,
        metadata={"help": "If training from scratch, pass a True here"},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

        assert self.generate_target is None, "generate_target argument has no effect in pre-training!"

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # dataset_name: Optional[str] = field(
    #     default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    # )
    # dataset_config_name: Optional[str] = field(
    #     default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    # )
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None,
                                      metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    id2entity_file: str = field(
        default=None,
        metadata={"help": "The path to the id2entity file."},
    )
    already_tokenized: Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, it means that the above files are already tokenized"},
    )
    already_processed: Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, then we don't need to scan the dataset and can directly call load_from_disk."
                          "The path passed by `dataset_dir` will be used as data source."},
    )
    process_data_only: Optional[bool] = field(
        default=False,
        metadata={"help": "If set to True, then only process data and don't do training or eval."
                          "Has higher priority than do_train and do_eval."
                          "Only effective when already_processed is `False` and save_processed_data is `True`."},
    )
    save_processed_data: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to save the processed data files for further use. "
                          "Only effective if already_processed is `False`."},
    )
    processed_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the processed data files."
                          "Only effective if already_processed is `False`."},
    )
    datasets_inmemory_maxsize: Optional[float] = field(
        default=0.0,
        metadata={"help": "The maximum memory size that the datasets library can use to store data"},
    )
    # overwrite_cache: bool = field(
    #     default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    # )
    validation_split_percentage: Optional[float] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    do_mlm: bool = field(
        default=True, metadata={"help": "Set to `True` if random masking is performed in pre-training."}
    )
    do_ssm: bool = field(
        default=True, metadata={"help": "Set to `True` if salient span masking is performed in pre-training."}
    )
    mlm_probability: float = field(
        default=0.1, metadata={"help": "Ratio of tokens to mask for masked language modeling"}
    )
    ssm_probability: float = field(
        default=0.2, metadata={"help": "Ratio of tokens to mask for salient span masking"}
    )
    mlm_random_replace: float = field(
        default=0.0, metadata={"help": "Ratio of `masked` tokens being replaced by random token instead of [MASK]"}
    )
    loss_on_mask_tokens: bool = field(
        default=False, metadata={"help": "Set to `True` if LM loss only apply on mask tokens. "
                                         "Otherwise, loss in apply on all output tokens."}
    )
    # line_by_line: bool = field(
    #     default=False,
    #     metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    # )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    add_another_bos: bool = field(
        default=True,
        metadata={
            "help": "Add another BOS token at the beginning of input sequences."
        },
    )
    enable_mpi: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Add another BOS token at the beginning of input sequences."
        },
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory containing model checkpoint (for evaluation)."
        },
    )

    def __post_init__(self):
        # if self.dataset_name is None and self.train_file is None and self.validation_file is None:
        #     raise ValueError("Need either a dataset name or a training/validation file.")
        if self.train_file is None and self.validation_file is None and self.dataset_dir is None:
            raise ValueError("Need to provide at least one of these: train_file, validation_file or dataset_dir")
        if self.dataset_dir is not None:
            assert self.train_file is None and self.validation_file is None, \
                "No need to specify train_file and validation_file if already specified dataset_dir"
        if self.train_file is not None and self.validation_file is not None:
            assert self.dataset_dir is None, \
                "No need to specify dataset_dir if already specified train_file and validation_file"
        if self.process_data_only:
            assert self.save_processed_data and self.processed_data_dir is not None
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in ["csv", "json",
        #                              "txt"], "`validation_file` should be a csv, a json or a txt file."

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
