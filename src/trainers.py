import collections
import inspect
import math
import sys
import os
import re
import json
import shutil
import functools
import time
import warnings
from pathlib import Path
import importlib.util
from packaging import version
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging, is_sagemaker_mp_enabled
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    TrainOutput,
    ShardedDDPOption,
    set_seed,
    speed_metrics,
    FSDPOption,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_torch_tpu_available,
)
from transformers.deepspeed import (
    deepspeed_init,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
    get_module_class_from_name
)

from transformers.utils import logging
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

# from transformers.trainer import _model_unwrap
from transformers.optimization import Adafactor, AdamW, get_scheduler
import copy
import random

from xnli_prompt import XNLI_FEW_SHOTS
from paws_x_prompt import PAWSX_FEW_SHOTS
from xstorycloze_prompt import XSTORYCLOZE_FEW_SHOTS
from xcopa_prompt import XCOPA_FEW_SHOTS
from xwinograd_prompt import XWINOGRAD_FEW_SHOTS

_NAME2CLASS={
    "xnli":XNLI_FEW_SHOTS,
    "xcopa":XCOPA_FEW_SHOTS,
    "xstorycloze":XSTORYCLOZE_FEW_SHOTS,
    "xwinograd":XWINOGRAD_FEW_SHOTS,
    "pawsx":PAWSX_FEW_SHOTS,
}


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

import numpy as np
from datetime import datetime
from filelock import FileLock

logger = logging.get_logger(__name__)


def get_prompt_format(prompt_formats_path, prompt_formats_name, prompt_language:str="en"):
    prompt_format = None
    if prompt_formats_path is not None:
        with open(prompt_formats_path,"r") as f:
            formats = json.load(f)
            prompt_format = formats[prompt_formats_name]
        # language is first selected
        prompt_format = prompt_format[prompt_language]
    return prompt_format

def batch_prompts_inference_choices(
        model, 
        tokenizer, 
        sample_dicts:list,
        prompt_style:str="{demos}{premise},right?{choice},{hypothesis}",
        choice_strs=[["entailment", "neutral", "contradiction"]],
        cuda=True, 
    ):
    """
    To inference the following prompt styles defined in batch level:
    1. {demos}{premise},right?[Choice_str],{hypothesis}
    2. {demos}Premise{premise}\nHypothesis: {hypothesis}\nQuestion: What is the relationship between premise and hypothesis, entailment, neutral, or contradiction?\nAnswer: {Choice_str}
    """
    batch_size = len(sample_dicts)
    num_choice = len(choice_strs[0])
    infer_sents = []
    for s_i, sample_dict in enumerate(sample_dicts):
        assert "choice" not in sample_dict, f"The key choice is used to insert candidate answer."
        
        for s in choice_strs[s_i]:
            sample_dict['choice'] = s
            infer_sents.append(prompt_style.format_map(sample_dict))
        sample_dict.pop('choice')

    tokenizer.pad_token = tokenizer.eos_token

    infer_dicts = tokenizer(
        text=infer_sents,
        padding="longest",
        return_tensors="pt",
    )

    # print(f"Current choice_ids for generation is {choice_ids}.")

    if cuda:
        model = model.cuda()
        for k in infer_dicts.keys():
            infer_dicts[k] = infer_dicts[k].cuda()

    with torch.inference_mode():
        # print(f"prompt_ids: {prompt_ids}")
        # out = model(prompt_ids, use_cache=True, return_dict=True)
        # out = model(**infer_dicts, return_dict=True)
        # CL-Model!
        out = model.clm_forward(**infer_dicts, return_dict=True)
        # past_key_values = out.past_key_values
        # print(f"past_key_values: {past_key_values}")
        # last_token_logits = out.logits[0][-1]
        last_layer_logits = out.logits.to(torch.float32)

        shift_labels = infer_dicts["input_ids"][..., 1:].contiguous()
        shift_last_logits = last_layer_logits[..., :-1, :].contiguous()
        shift_score_mask = infer_dicts["attention_mask"].unsqueeze(-1)[..., 1:, :].contiguous()
        
        # (batch_size*num_choice, max_seq_len-1, vocab_size)
        shift_labels = shift_labels.unsqueeze(-1).expand_as(shift_last_logits)
        # (batch_size*num_choice, max_seq_len-1, vocab_size)
        last_layer_scores = torch.log2(torch.softmax(shift_last_logits,-1)) * shift_score_mask
        # (batch_size, num_choice)
        last_layer_scores = torch.sum(torch.gather(last_layer_scores, -1, shift_labels)[..., 0], -1).view(batch_size, num_choice)

        # print(f"{sample_dicts}, scores are {last_layer_scores}")
        res = torch.argmax(last_layer_scores, -1).tolist()

    return res

def print2log(samples, inference_time, num_samples, num_shots, acc:float, out_handle):
    print(f"Demos are: \n{samples.demos_str}\n*******")
    print(f"Prompt style is: \n{samples.FEW_SHOT_FORMAT}\n*******")
    print(f"Inference time is {inference_time:.2f}s for {num_samples} samples")
    print(f"Accuracy = {acc:.8f} ({num_shots}-shots eval, and {num_samples} test samples)")

    # out_handle.write(f"\nInference prompt string is: \n{xnli_samples.samples_str}\n*******")
    out_handle.write(f"\nDemos are: \n{samples.demos_str}\n*******")
    out_handle.write(f"\nPrompt style is: \n{samples.FEW_SHOT_FORMAT}\n*******")
    out_handle.write(f"\nInference time is {inference_time:.2f}s for {num_samples} samples")
    out_handle.write(f"\nAccuracy = {acc:.8f} ({num_shots}-shots eval, and {num_samples} test samples)")

def avg_list(l_in):
    return sum(l_in) / float(len(l_in))


class CLTrainer(Trainer):

    def _evaluate_loop(
        self,
        samples, 
        batch_size
    ):
        model = self.model
        tokenizer = self.tokenizer

        preds = []
        ans = []

        curr_dicts = []
        curr_choices = []
        
        for sample in samples:
            curr_dicts.append(sample["sample"])
            curr_choices.append(samples.choices(sample))

            gold = samples.choice_idx(sample)
            ans.append(gold)

            if len(curr_dicts) < batch_size and len(ans)<len(samples):
                continue
            
            pred_batch = batch_prompts_inference_choices(
                model,
                tokenizer,
                sample_dicts=curr_dicts,
                prompt_style=samples.FEW_SHOT_FORMAT,
                choice_strs=curr_choices,
            )

            preds.extend(pred_batch)

            curr_dicts = []
            curr_choices = []
        
        acc = sum(1 for x,y in zip(preds, ans) if x == y) / len(ans)

        return acc, preds, ans

    def evaluate_xlang_datasets(
        self,
        batch_size,
        seed:int,
        dataset_name:str,
        prompt_sample_path:str,
        test_sample_path:str,
        num_shots:str,
        prompt_language:str,
        prompt_format,
    ):
        """
        evaluate the few shot performance in-context learning
        """
        
        samples = _NAME2CLASS[dataset_name](
            prompt_sample_path=prompt_sample_path,
            test_sample_path=test_sample_path,
            num_shots=num_shots,
            seed=seed,
            prompt_language=prompt_language,
            prompt_format=prompt_format,
            )

        # inference with few shots and generate answers by hidden states for A, B or C
        # s_t = time.time()

        acc, preds, ans = self._evaluate_loop(
            samples=samples,
            batch_size=batch_size
        )

        return acc, preds, ans

    def evaluate_few_shots(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:
        self.model.eval()
        metrics = {}

        with open(self.args.eval_config_file, "r") as f:
            eval_configs = json.load(f)
        
        for cfg in eval_configs:
            cfg_dict = {"data_dir": cfg["data_dir"]}

            prompt_format = None
            if "prompt_formats_path" in cfg:
                with open(cfg["prompt_formats_path"].format_map(cfg_dict),"r") as f:
                    formats = json.load(f)
                    prompt_format = formats[cfg["prompt_formats_name"]]
                
            for s_i, sample_language in enumerate(cfg["sample_languages"]):
                prompt_language = cfg["prompt_languages"][s_i]
                cfg_dict["prompt_language"] = prompt_language
                cfg_dict["sample_language"] = sample_language

                # language is first selected
                curr_prompt_format = prompt_format[prompt_language] if prompt_format is not None else None
                
                for num_shot in cfg["num_shots"]:
                    for inf_i, infer_seed in enumerate(cfg["infer_seeds"]):
                        self.log({"INFO-STR":f"Now evaluate dataset-{cfg['dataset_name']} in {num_shot}-shots using {sample_language}-samples and {prompt_language}-prompt (infer-seed:{infer_seed})"})
                        
                        acc, preds, ans, = self.evaluate_xlang_datasets(
                            batch_size=self.args.per_device_eval_batch_size,
                            seed=infer_seed,
                            num_shots=num_shot,
                            dataset_name=cfg["dataset_name"],
                            prompt_sample_path=cfg["prompt_sample_path"].format_map(cfg_dict),
                            test_sample_path=cfg["test_sample_path"].format_map(cfg_dict),
                            prompt_language=prompt_language,
                            prompt_format=curr_prompt_format,
                        )

                        self.log({"INFO-STR":f"Evaluation is done! Accuracy is {acc:.4f}"})

                        metrics[f"{cfg['dataset_name']}_{sample_language}_s{num_shot}_i{infer_seed}"] = acc

                        if num_shot == 0 and inf_i >0:
                            # Same result for all seeds in zero-shot settings
                            break
        
        # the average of all results
        avg_all = avg_list([v for k,v in metrics.items()])
        # the average of all zero-shots
        res_keys = ["_s0_"]
        avg_zero = avg_list([v for k,v in metrics.items() if all(rk in k for rk in res_keys)])

        metrics["eval_avg_all"] = avg_all
        metrics["eval_avg_zero"] = avg_zero

        self.log(metrics)
        return metrics

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:
        
        metrics = self.evaluate_few_shots(eval_dataset, ignore_keys, metric_key_prefix, eval_senteval_transfer)

        return metrics


    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """

        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save.
        # assert _model_unwrap(model) is self.model, "internal model should be a reference to self.model"

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                output_dir = self.args.output_dir
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                # Only save model when it is the best one
                self.save_model(output_dir)
                if self.deepspeed:
                    self.deepspeed.save_checkpoint(output_dir)

                # Save optimizer and scheduler
                if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                    self.optimizer.consolidate_state_dict()

                if is_torch_tpu_available():
                    xm.rendezvous("saving_optimizer_states")
                    xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        reissue_pt_warnings(caught_warnings)
                elif self.is_world_process_zero() and not self.deepspeed:
                    # deepspeed.save_checkpoint above saves model/optim/sched
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)

                # Save the Trainer state
                if self.is_world_process_zero():
                    self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        else:
            # Save model checkpoint
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            if self.deepspeed:
                # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
                # config `stage3_gather_16bit_weights_on_model_save` is True
                self.deepspeed.save_checkpoint(output_dir)

            # Save optimizer and scheduler
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer.consolidate_state_dict()

            if is_torch_tpu_available():
                xm.rendezvous("saving_optimizer_states")
                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                    reissue_pt_warnings(caught_warnings)
            elif self.args.should_save and not self.deepspeed:
                # deepspeed.save_checkpoint above saves model/optim/sched
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save:
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            # Save RNG state in non-distributed training
            rng_states = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
            }
            if torch.cuda.is_available():
                if self.args.local_rank == -1:
                    # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                    rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
                else:
                    rng_states["cuda"] = torch.cuda.random.get_rng_state()

            if is_torch_tpu_available():
                rng_states["xla"] = xm.get_rng_state()

            # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
            # not yet exist.
            os.makedirs(output_dir, exist_ok=True)

            if self.args.world_size <= 1:
                torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
            else:
                torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
    
    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        
        The main difference between ours and Huggingface's original implementation is that we 
        also load model_args when reloading best checkpoints for evaluation.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)
            if not self.is_model_parallel:
                model = model.to(self.args.device)

            self.model = model
            self.model_wrapped = model

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps


        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )

        if self.args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=None
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine  # will get further wrapped in DDP
            self.deepspeed = deepspeed_engine  # DeepSpeedEngine object
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        model = self.model_wrapped

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        self.sharded_ddp = None
        if self.fsdp is not None:
            if not self.args.fsdp_config["xla"]:
                # PyTorch FSDP!
                from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

                if FSDPOption.OFFLOAD in self.args.fsdp:
                    cpu_offload = CPUOffload(offload_params=True)
                else:
                    cpu_offload = CPUOffload(offload_params=False)

                auto_wrap_policy = None

                if FSDPOption.AUTO_WRAP in self.args.fsdp:
                    if self.args.fsdp_config["fsdp_min_num_params"] > 0:
                        auto_wrap_policy = functools.partial(
                            size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config["fsdp_min_num_params"]
                        )
                    elif self.args.fsdp_config.get("fsdp_transformer_layer_cls_to_wrap", None) is not None:
                        transformer_cls_to_wrap = set()
                        for layer_class in self.args.fsdp_config["fsdp_transformer_layer_cls_to_wrap"]:
                            transformer_cls = get_module_class_from_name(model, layer_class)
                            if transformer_cls is None:
                                raise Exception("Could not find the transformer layer class to wrap in the model.")
                            else:
                                transformer_cls_to_wrap.add(transformer_cls)
                        auto_wrap_policy = functools.partial(
                            transformer_auto_wrap_policy,
                            # Transformer layer class to wrap
                            transformer_layer_cls=transformer_cls_to_wrap,
                        )
                mixed_precision_policy = None
                dtype = None
                if self.args.fp16:
                    dtype = torch.float16
                elif self.args.bf16:
                    dtype = torch.bfloat16
                if dtype is not None:
                    mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
                if type(model) != FSDP:
                    # XXX: Breaking the self.model convention but I see no way around it for now.
                    self.model = model = FSDP(
                        model,
                        sharding_strategy=self.fsdp,
                        cpu_offload=cpu_offload,
                        auto_wrap_policy=auto_wrap_policy,
                        mixed_precision=mixed_precision_policy,
                        device_id=self.args.device,
                        backward_prefetch=self.backward_prefetch,
                        forward_prefetch=self.forword_prefetch,
                        limit_all_gathers=self.limit_all_gathers,
                    )
        elif self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)


        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(train_dataloader) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            assert train_dataset_is_sized, "currently we only support sized dataloader!"

            inputs = None
            last_inputs = None
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self._total_flos += self.floating_point_ops(inputs)

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and self.args.fp16:
                            self.optimizer.clip_master_grads(self.args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval=None)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval=None)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint, model_args=self.model_args)
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)