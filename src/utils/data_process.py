
import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import wandb

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

from transformers import(
    T5Tokenizer,
    
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
)


# 'task_dataset', 'task_type', 
# 'answer_choice' : null or list[str]


your_data_path="datasets/PromptCBLUE"

train_file =  os.path.join(your_data_path, 'train.json')
validation_file =  os.path.join(your_data_path, 'dev.json')
test_file =  os.path.join(your_data_path, 'test.json')
# Load dataset
data_files = {}
if train_file is not None:
    data_files["train"] = train_file
    extension = train_file.split(".")[-1]
if validation_file is not None:
    data_files["validation"] = validation_file
    extension = validation_file.split(".")[-1]
if test_file is not None:
    data_files["test"] = test_file
    extension = test_file.split(".")[-1]

lm_datasets = load_dataset(
    extension,
    data_files=data_files,
)


print(list(set(lm_datasets['train']['task_type'])))
# hash = {}
# for task in lm_datasets['train']['task_dataset']:
#     if task not in hash:
#         hash[task] = 1
#     else:
#         hash[task] += 1
# print(hash)


def split_task(dataset, dim):
    dim = 'validation'
    task_dataset = {}
    for i, task in enumerate(dataset[dim]['task_dataset']):
        if task not in task_dataset:
            task_dataset[task] = [dataset[dim][i]]
        else:
            task_dataset[task].append(dataset[dim][i])
    for task in task_dataset.keys():
        if not os.path.exists(f'./data/{task}'):
            os.mkdir(f'./data/{task}')
        with open(os.path.join(f'./data/{task}', f'dev.json'), 'w',encoding='utf-8') as file:
            json.dump(task_dataset[task], file, ensure_ascii=False)