import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
import argparse
import time
import copy
import datetime
from glob import glob
from tqdm import tqdm,trange
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,classification_report, confusion_matrix
from transformers import RobertaTokenizerFast, AutoConfig, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader,TensorDataset
from masker import BertMasker

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

def tokenize(args, data_path, tokenizer):
    text_labels = []
    text_inputs = []
    dialog_ids = []
    turn_ids = []
    total_data = list()

    with open(data_path, "r") as file_id:
        _raw_data = json.load(file_id)
        num_utterances = 2 * args.max_turns + 1
        num_instances = len(_raw_data)
        for data in tqdm(_raw_data, desc="Tokenize"):
            dialog_datum = data
            dialog = data['input_text'].copy()
            for turn_id, turn in enumerate(dialog):
                if turn_id % 2 == 0:
                    dialog[turn_id] = "<USER> " + turn
                else:
                    dialog[turn_id] = "<SYS> " + turn
            text = " ".join(dialog[-num_utterances :])
            text_inputs.append(text)
            text_labels.append(dialog_datum["disambiguation_label_gt"])
            dialog_ids.append(dialog_datum["dialog_id"])
            turn_ids.append(dialog_datum["turn_id"])

        encoded_inputs = tokenizer(
            text_inputs, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True,
        )
        args.total_tokens = torch.nonzero(encoded_inputs['attention_mask']==1)

    return encoded_inputs,torch.tensor(text_labels, dtype=torch.long)


def tokenize_without_additional_tokens(args, data_path, tokenizer):
    text_labels = []
    text_inputs = []
    dialog_ids = []
    turn_ids = []
    total_data = list()

    with open(data_path, "r") as file_id:
        _raw_data = json.load(file_id)

        num_utterances = 2 * args.max_turns + 1
        num_instances = len(_raw_data)
        for data in tqdm(_raw_data, desc="Tokenize"):
            dialog_datum = data
            dialog = data['input_text'].copy()
            for turn_id, turn in enumerate(dialog):
                # if turn_id % 2 == 0:
                #     dialog[turn_id] = "<USER> " + turn
                # else:
                #     dialog[turn_id] = "<SYS> " + turn
                if turn_id == 0:    dialog[turn_id] = turn
                else:               dialog[turn_id] = "</s><s>" + turn                

            dialog = dialog[-num_utterances :]
            dialog[0] = dialog[0].replace("</s><s>", "")
            text = "".join(dialog)
            text_inputs.append(text)
            text_labels.append(dialog_datum["disambiguation_label_gt"])
            dialog_ids.append(dialog_datum["dialog_id"])
            turn_ids.append(dialog_datum["turn_id"])
        # print("\n==================================================")
        # print("4 _raw_data : {}".format(_raw_data[0]['input_text']))            
        # print("Text inputs : {}".format(text_inputs[0]))
        # import sys
        # sys.exit()        
        encoded_inputs = tokenizer(
            text_inputs, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True,
        )
        args.total_tokens = torch.nonzero(encoded_inputs['attention_mask']==1)
    return encoded_inputs,torch.tensor(text_labels, dtype=torch.long)


def tokenize_split_history_and_current(args, data_path, tokenizer):
    text_labels = []
    text_inputs = []
    dialog_ids = []
    turn_ids = []
    total_data = list()

    with open(data_path, "r") as file_id:
        _raw_data = json.load(file_id)
        num_utterances = 2 * args.max_turns + 1
        num_instances = len(_raw_data)
        for data in tqdm(_raw_data, desc="Tokenize"):
            dialog_datum = data
            dialog = data['input_text'].copy()

            delimiter = ""
            for turn_id, turn in enumerate(dialog):
                if turn_id+1 ==len(dialog)-1:   # 마지막 발화 턴이라면
                    delimiter = "</s><s> "

                if turn_id % 2 == 0:
                    dialog[turn_id] = delimiter + "<USER> " + turn
                else:
                    dialog[turn_id] = "<SYS> " + turn

            dialog = dialog[-num_utterances :]
            dialog[0] = dialog[0].replace("</s><s>", "")
            text = " ".join(dialog)
            text_inputs.append(text)
            text_labels.append(dialog_datum["disambiguation_label_gt"])
            dialog_ids.append(dialog_datum["dialog_id"])
            turn_ids.append(dialog_datum["turn_id"])

        encoded_inputs = tokenizer(
            text_inputs, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True,
        )
        args.total_tokens = torch.nonzero(encoded_inputs['attention_mask']==1)
    return encoded_inputs,torch.tensor(text_labels, dtype=torch.long)


def get_masker(args, tokenizer):
    if args.mode == 'fine': return None
    elif not args.mode == 'fine' : return BertMasker(tokenizer)

def idx_to_tensor(args, data, labels, train=False):

    dataset  = TensorDataset(data.input_ids, data.attention_mask, labels)
    dataloader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=train)

    return dataloader

def save_checkpoint(args, best_model):
    today = str(datetime.datetime.today().strftime("%Y%m%d"))

    if not args.mode == 'post': output_dir = "./save_model/{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}".format(today, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)
    if args.mode == 'post': output_dir = "./save_model/post/{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}".format(today, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)                  
    model = best_model.module if hasattr(best_model, "module") else best_model    

    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.bin"))

def convert_batch_to_inputs(args, batch, masker):

    if args.mode == 'fine':
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
                "input_ids" : batch[0], 
                "attention_mask" : batch[1], 
                "labels" : batch[2]}        
    elif args.mode == 'post':
        batch[0], masked_labels = masker.mask_tokens(batch[0], args.mlm_probability)        
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
                "input_ids" : batch[0], 
                "attention_mask" : batch[1], 
                "labels" : masked_labels.to(args.device)}        

    elif args.mode == 'multi':
        batch[0], masked_labels = masker.mask_tokens(batch[0], args.mlm_probability)        
        batch = tuple(t.to(args.device) for t in batch)

        inputs = {
                "args" : args,
                "input_ids" : batch[0], 
                "attention_mask" : batch[1], 
                "labels" : batch[2],
                "masked_lm_labels" : masked_labels.to(args.device)}        

    return inputs

def default_parser(parser):
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", type=str, default='fine')
    parser.add_argument("--model", type=str, default='roberta-base')
    parser.add_argument("--model_name_or_path", type=str, default=None)

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)

    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_proportion", type=float, default=0.06)

    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--mlm_loss", type=float, default=0.2)

    parser.add_argument("--train_file", type=str, default="./data/")
    parser.add_argument("--dev_file", type=str, default="./data/")
    parser.add_argument("--devtest_file", type=str, default="./data/")

    parser.add_argument("--train", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--max_turns", type=int, default=5)

    parser.add_argument("--total_tokens", type=int, default=0)

    return parser