# #!/usr/bin/env python3

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.

Adapted from:
https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py
"""


import argparse
import glob
import json
import logging
import os
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import BartForConditionalGeneration as BartLMHeadModel

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


import tensorboardX
from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TextDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )

        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        for i in range(
            0, len(tokenized_text) - block_size + 1, block_size
        ):  # Truncate in block of block_size
            self.examples.append(
                tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i : i + block_size]
                )
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, pred_file_path: str, target_file_path: str, block_size=512
    ):
        assert os.path.isfile(pred_file_path)
        assert os.path.isfile(target_file_path)

        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", pred_file_path)
        logger.info("Creating features from dataset file at %s", target_file_path)

        with open(pred_file_path, encoding="utf-8") as f:
            pred_lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]

        with open(target_file_path, encoding="utf-8") as f:
            target_lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]

        input_ = tokenizer.batch_encode_plus(
            pred_lines, add_special_tokens=True, max_length=block_size, truncation=True
        )
        output_ = tokenizer.batch_encode_plus(
            target_lines, add_special_tokens=True, max_length=block_size, truncation=True
        )
        self.src = input_["input_ids"]
        self.src_mask = input_["attention_mask"]
        self.tgt = output_["input_ids"]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return torch.tensor(self.src[i], dtype=torch.long), torch.tensor(self.src_mask[i], dtype=torch.long), torch.tensor(self.tgt[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    pred_file_path = args.eval_data_pred_file if evaluate else args.train_data_pred_file
    target_file_path = args.eval_data_target_file if evaluate else args.train_data_target_file

    if args.line_by_line:
        dataset = LineByLineTextDataset(
            tokenizer, args, pred_file_path=pred_file_path, target_file_path=target_file_path, block_size=args.block_size
        )
    else:
        dataset = TextDataset(
            tokenizer, args, file_path=file_path, block_size=args.block_size
        )

    # Unknown issues have been reported around not being able to handle incomplete batches (e.g. w/ older CUDA 9.2)
    # Below is a workaround in case you encounter this issue.
    # Alternatively, --nocuda could avoid this issue too.
    # Comment out the following if you do not encounuter this issue or if you are not using any GPU.
    n = len(dataset) % args.per_gpu_train_batch_size
    if n != 0:
        print("Truncating from %d examples" % len(dataset.src))
        dataset.src = dataset.src[:-n]
        dataset.src_mask = dataset.src_mask[:-n]
        dataset.tgt = dataset.tgt[:-n]
        print("Truncating to %d examples" % len(dataset.src))
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_device(args):
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        args.device = torch.device('cpu')
        print('Device name: cpu')

def _sorted_checkpoints(
    args, checkpoint_prefix="checkpoint", use_mtime=False
) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(
        os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix))
    )

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                checkpoint
            )
        )
        shutil.rmtree(checkpoint)


def train(
    args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> Tuple[int, float]:
    """Train the model"""

    args.train_batch_size = args.per_gpu_train_batch_size
    tb_writer = SummaryWriter()

    def collate(examples):
        # src_list = list(map(lambda x: x[0], examples))
        # src_mask_list = list(map(lambda x: x[1], examples))
        src_list = list(map(lambda x: x[0], examples))
        src_mask_list = list(map(lambda x: x[1], examples))

        if tokenizer._pad_token is None:
            src_pad = pad_sequence(src_list, batch_first=True)
        else:
            src_pad = pad_sequence(src_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        src_mask_pad = pad_sequence(src_mask_list, batch_first=True, padding_value=0)

        if len(examples[0]) == 2:        
            return src_pad, src_mask_pad

        tgt_list = list(map(lambda x: x[2], examples))

        if tokenizer._pad_token is None:
            tgt_pad = pad_sequence(tgt_list, batch_first=True)
        else:
            tgt_pad = pad_sequence(tgt_list, batch_first=True, padding_value=tokenizer.pad_token_id)

        return src_pad, src_mask_pad, tgt_pad

    train_sampler = (
        RandomSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    model = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    # if args.fp16:
    #     # try:
    #     #     from apex import amp
    #     # except ImportError:
    #     #     raise ImportError(
    #     #         "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
    #     #     )
    #     model, optimizer = torch.cuda.amp.initialize(
    #         model, optimizer, opt_level=args.fp16_opt_level,
    #     )

    # multi-gpu training (should be after apex fp16 initialization)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * 1,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint

    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            # inputs, labels = (
            #     mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            # )

            src = batch[0].to(args.device)
            src_mask = batch[1].to(args.device)
            tgt = batch[2].to(args.device)

            # print("\nSrc : {} {}".format(src.size(), src[0]))
            # print("Src Mask : {} {}".format(src_mask.size(), src_mask[0]))
            # print("Tgt : {} {}".format(tgt.size(), tgt[0]))
            model.train()    

            # import sys
            # sys.exit(0)
            loss = model(input_ids=src, attention_mask=src_mask, labels=tgt)[0]
            # print("{} step : {}".format(step, loss))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule                
                model.zero_grad()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                global_step += 1

                if (
                    args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate evaluate_during_trainingwhen single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                    args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "{}-{}".format(checkpoint_prefix, global_step)
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(
    args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix=""
) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly

    def collate(examples):
        # src_list = list(map(lambda x: x[0], examples))
        # src_mask_list = list(map(lambda x: x[1], examples))
        src_list = list(map(lambda x: x[0], examples))
        src_mask_list = list(map(lambda x: x[1], examples))

        if tokenizer._pad_token is None:
            src_pad = pad_sequence(src_list, batch_first=True)
        else:
            src_pad = pad_sequence(src_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        src_mask_pad = pad_sequence(src_mask_list, batch_first=True, padding_value=0)

        if len(examples[0]) == 2:        
            return src_pad, src_mask_pad

        tgt_list = list(map(lambda x: x[2], examples))

        if tokenizer._pad_token is None:
            tgt_pad = pad_sequence(tgt_list, batch_first=True)
        else:
            tgt_pad = pad_sequence(tgt_list, batch_first=True, padding_value=tokenizer.pad_token_id)

        return src_pad, src_mask_pad, tgt_pad
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate,
    )



    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # inputs, labels = (
        #     mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        # )
        # inputs = inputs.to(args.device)
        # labels = labels.to(args.device)
        src = batch[0].to(args.device)
        src_mask = batch[1].to(args.device)
        tgt = batch[2].to(args.device)
        lm_loss = None
        with torch.no_grad():
            lm_loss = model(input_ids=src, attention_mask=src_mask, labels=tgt)[0]
            # print("\nlm_loss : {}".format(lm_loss))
            eval_loss += lm_loss.mean().item()

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    # print("eval_loss : {}".format(eval_loss))

    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_pred_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--train_data_target_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_pred_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--eval_data_target_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )

    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue",
        action="store_true",
        help="Whether to continue from latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--add_special_tokens",
        default=None,
        type=str,
        help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    args = parser.parse_args()

    # set random seed from beginning
    set_seed(args)

    if (
        args.model_type in ["bert", "roberta", "distilbert", "camembert"]
        and not args.mlm
    ):
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_target_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_target_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError(
                "Used --should_continue but no checkpoint was found in --output_dir."
            )
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )


    # Setup CUDA, GPU & distributed training
    set_device(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        1,
        bool(args.local_rank != -1),
        args.fp16,
    )


    if args.config_name:
        print("Config name : {}".format(args.config_name))
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        print("model_name_or_path : {}".format(args.model_name_or_path))
        
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, cache_dir=args.cache_dir
        )
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    tokenizer = ""
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, cache_dir=args.cache_dir
        )

    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, cache_dir=args.cache_dir
        )

    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    tokenizer.eos_token = '<end>'
    tokenizer.cls_token = '<cls>'
    tokenizer.unk_token = '<unk>'
    tokenizer.pad_token = '<pad>'        


    if args.add_special_tokens:
        if not os.path.exists(args.add_special_tokens):
            raise ValueError(
                "Additional special tokens file {args.add_special_tokens} not found}"
            )
        with open(args.add_special_tokens, "rb") as handle:
            special_tokens_dict = json.load(handle)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} tokens")
        logger.info(f"All special tokens: {tokenizer.all_special_tokens}")

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    if args.model_name_or_path:
        if 'bart' in args.model_name_or_path:
            model = BartLMHeadModel.from_pretrained(
                args.model_name_or_path, 
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config, 
                cache_dir=args.cache_dir
                )
        else:
            model = AutoModelWithLMHead.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            ) 
    else:
        logger.info("Training new model from scratch")


        if 'bart' in args.model_name_or_path:    model = BartLMHeadModel.from_config(config)            
        else:                                    model = AutoModelWithLMHead.from_config(config)

    # ensure model aligns with any addition of special tokens
    # (unclear if this step is needed for a new model)
    if args.add_special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        tokenizer.eos_token = '<end>'
        tokenizer.cls_token = '<cls>'
        tokenizer.unk_token = '<unk>'
        tokenizer.pad_token = '<pad>'        
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = [
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            ]
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = (
                checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            )

            model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = {k + "_{}".format(global_step): v for k, v in result.items()}
            results.update(result)

    return results


if __name__ == "__main__":
    main()
