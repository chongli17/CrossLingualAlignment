import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random
from itertools import chain

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    XGLMForCausalLM,
    BloomForCausalLM,
    LlamaForCausalLM,
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available

from decoder_models import GPT2ForCL, GPTNeoForCL, XGLMForCL, BloomForCL, LlamaForCL

from trainers import CLTrainer

# offline wandb runs and increase the waiting time for wandb
# os.environ["WANDB_MODE"]="offline"
# os.environ["WANDB__SERVICE_WAIT"] = "300"


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

_IGNORE_INDEX = -100

_MODEL_TYPE2CLASS = {
    "gpt2": GPT2ForCL,
    "gpt_neo": GPTNeoForCL,
    "xglm": XGLMForCL,
    "bloom": BloomForCL,
    "llama": LlamaForCL
}

_CLM_MODEL = {
    "gpt2": GPT2LMHeadModel,
    "gpt_neo": GPTNeoForCausalLM,
    "xglm": XGLMForCausalLM,
    "bloom": BloomForCausalLM,
    "llama": LlamaForCausalLM,
}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
        default=True,
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

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg-*, avg_bottom, avg_top2, avg_bottom2, avg_first_last, last, last-*, last_top2, last_first_last, max-*)."
        }
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    do_clm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use CLM auxiliary objective."
        }
    )
    clm_weight: float = field(
        default=1.5,
        metadata={
            "help": "Weight for CLM auxiliary objective (only effective if --do_clm)."
        }
    )
    cl_weight: float = field(
        default=1,
        metadata={
            "help": "Weight for CL auxiliary objective (only effective if --do_clm)."
        }
    )
    cl_mlp: bool = field(
        default=False,
        metadata={
            "help": "Use MLP on the sentence representation."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    clm_block_size: Optional[int] = field(
        default=None, 
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset for clm will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    eval_config_file: str = field(
        default="",
        metadata={"help": "Path to json config file for eval the cross-language dataset."}
    )

    @cached_property
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from transformers.deepspeed import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    logger.info("Initial Parameters:")
    # wandb.config.update(model_args)
    for attr, value in sorted(model_args.__dict__.items()):
        logger.info(f"\t{attr}={value}")
    # wandb.config.update(data_args)
    for attr, value in sorted(data_args.__dict__.items()):
        logger.info(f"\t{attr}={value}")
    # wandb.config.update(training_args)
    for attr, value in sorted(training_args.__dict__.items()):
        logger.info(f"\t{attr}={value}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/cache/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/cache/")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
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
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

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

    # set pad_token
    if config.model_type == "gpt2" or config.model_type == "gpt_neo" or config.model_type == "llama":
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id

    if model_args.model_name_or_path:
        model = _MODEL_TYPE2CLASS[config.model_type].from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args                  
            )
        
        if config.model_type == "bert" and model_args.do_mlm:
            pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
            model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())

        if model_args.do_clm:
            pretrained_model = _CLM_MODEL[config.model_type].from_pretrained(model_args.model_name_or_path)
            model.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())
            

    else:
        raise NotImplementedError

    # Prepare features
    clm_text_name = "clm_text"
    clm_prompt_len_name = "clm_prompt_len"
    column_names = datasets["train"].column_names.copy()
    if clm_text_name in column_names:
        column_names.pop(column_names.index(clm_text_name))
    if clm_prompt_len_name in column_names:
        column_names.pop(column_names.index(clm_prompt_len_name))

    sent2_cname = None
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        
        sentences = examples[sent0_cname] + examples[sent1_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
        
        # construct clm parallel samples
        if clm_text_name in list(examples.keys()):
            clm_sents = [ examples[clm_text_name][i] for i in range(total)]
            clm_prompts = [ examples[clm_text_name][i][:int(examples[clm_prompt_len_name][i])] for i in range(total)]
        else:
            clm_sents = [ examples[sent0_cname][i] + " = " + examples[sent1_cname][i] for i in range(total)]
            clm_prompts = [ examples[sent0_cname][i] + " = " for i in range(total)]
        
        clm_sents = clm_sents + clm_prompts

        clm_features = tokenizer(
            clm_sents,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding=False,
        )
        
        clm_prompt_len = []
        for clm_i in range(total):
            same_len = -1
            if examples[clm_prompt_len_name][clm_i] == 0:
                same_len = 0
            else:
                a_i = clm_features["input_ids"][clm_i]
                p_i = clm_features["input_ids"][clm_i+total]
                min_len = len(p_i) if len(p_i) <= len(a_i) else len(a_i)
                for t_j in range(min_len):
                    if a_i[t_j] != p_i[t_j]:
                        same_len = t_j
                        break
                same_len = same_len if same_len >=0 else min_len
            clm_prompt_len.append(same_len)

        features["clm_input_ids"] = clm_features["input_ids"][:total]
        # columns that not in model inputs will be removed
        features["clm_labels"] = clm_prompt_len
        return features

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability
        clm: bool = True

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = [
                'input_ids', 'attention_mask', 'token_type_ids', 
                'mlm_input_ids', 'mlm_labels', 
                'clm_input_ids', 'clm_labels'
            ]

            clm_input_ids = [{"input_ids": feature.pop("clm_input_ids"), "prompt_len": feature.pop("clm_labels")} for feature in features]

            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return

            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            if model_args.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])
            
            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            if model_args.do_clm:
                # batch['clm_input_ids'], batch['clm_labels'] = self.clm_tokens_block(flat_features, device=batch["input_ids"].device)
                batch['clm_input_ids'], batch['clm_labels'] = self.clm_tokens_block(clm_input_ids, device=batch["input_ids"].device)
                
            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch
        
        def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

        def clm_tokens(
            self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
        ):
            inputs = inputs.clone()
            labels = inputs.clone()
            # input_len = attention_mask.sum(-1).unsqueeze(-1)
            # for label, source_len in zip(labels, input_len):
            #     label[source_len:] = _IGNORE_INDEX
            labels = inputs * attention_mask + _IGNORE_INDEX*(1-attention_mask)
            return inputs, labels
        
        def clm_tokens_block(
            self, flat_features, device
        ):
            if data_args.clm_block_size is None:
                block_size = self.tokenizer.model_max_length
                if block_size > 1024:
                    logger.warning(
                        f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                        "Picking 1024 instead. You can change that default value by passing --block_size xxx."
                    )
                    block_size = 1024
            else:
                if data_args.clm_block_size > tokenizer.model_max_length:
                    logger.warning(
                        f"The block_size passed ({data_args.clm_block_size}) is larger than the maximum length for the model"
                        f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                    )
                block_size = min(data_args.clm_block_size, tokenizer.model_max_length)
            
            # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }

                # result["labels"] = result["input_ids"].copy()
                return result
            
            examples = {
                "input_ids": [flat_feature["input_ids"] for flat_feature in flat_features]
            }
            
            inputs = torch.tensor(group_texts(examples=examples)["input_ids"], device=device)

            label_ids = []
            for flat_feature in flat_features:
                flat_vec = torch.ones(len(flat_feature["input_ids"]),dtype=torch.int8)
                flat_vec[:flat_feature["prompt_len"]] = 0
                label_ids.append(flat_vec.tolist())
            
            labels = {
                "input_ids": label_ids
            }

            labels = torch.tensor(group_texts(examples=labels)["input_ids"], device=device)
            
            # Do not mask bos and eos
            labels = torch.logical_or(labels, (inputs == self.tokenizer.bos_token_id)).type(torch.int8)
            labels = torch.logical_or(labels, (inputs == self.tokenizer.eos_token_id)).type(torch.int8)

            labels = inputs*labels + _IGNORE_INDEX*(1-labels)

            # Change the last token for the row with [x, -100, ..., -100] to avoid nan loss
            for i in range(labels.shape[0]):
                if torch.sum(labels[i,1:] != _IGNORE_INDEX) == 0:
                    labels[i, -1] = inputs[i, -1]

            return inputs, labels #input_ids, labels

    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # results = trainer.evaluate(eval_senteval_transfer=True)
        results = trainer.evaluate(eval_senteval_transfer=False)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
