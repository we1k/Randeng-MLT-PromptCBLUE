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

from argparse import ArgumentParser

from transformers import (
    AutoConfig,
    AutoModel,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ["WANDB_MODE"]='disabled'

import sys
sys.path.append("./")

from peft import PeftConfig, LoraConfig, PeftModelForCausalLM, get_peft_model
from src.chatmed_llama_peft.instruction import TASK_TO_INSTRUCTION, TASK_TO_MAX_NEW_TOKENS

from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device


def main(args):
    # load_model 
    config = LlamaConfig.from_pretrained(
        args.model_name_or_path,
        # trust_remote_code=True
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        # trust_remote_code=True
    )

    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
    ).half().cuda()

    # load checkpoint
    LR=1e-4
    STEP=300
    peft_path = os.path.join('checkpoint',f'{args.task}-{LR}/checkpoint-{STEP}/adapter_model')
    print(peft_path)
    if not os.path.exists(peft_path):
        raise ValueError(f"No peft path :{peft_path}")

    peft_config = LoraConfig.from_pretrained(peft_path)
    model = get_peft_model(model, peft_config)
    model = PeftModelForCausalLM.from_pretrained(model, peft_path, is_trainable=False)
    model.print_trainable_parameters()
    
    
    # load dataset
    data_path="./datasets/toy_examples/"
    train_file =  os.path.join(data_path, 'train.json')
    validation_file =  os.path.join(data_path, 'dev.json')
    test_file =  os.path.join(data_path, 'test.json')
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

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
    )

    # Get the column names for input/target.
    prompt_column = 'input'
    response_column = 'target'

    column_names = raw_datasets["validation"].column_names
    # Temporarily set max_target_length for training.
    max_input_length = 1024
    
    
    def generate_prompt(instruction, data):
        return f"""### 指令:\n{instruction}\n### 输入:\n{data[prompt_column]}\n### 输出:\n{tokenizer.bos_token + data[response_column] + tokenizer.eos_token}
        """
        
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_input_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_input_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
    
        result["labels"] = result["input_ids"].copy()
 
        return result

    def preprocess_function(data_point):
        instruction = TASK_TO_INSTRUCTION[data_point['task_dataset']]
        full_prompt = generate_prompt(instruction, data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    # Main data processing function that will make each entry its own in the dataset
    def single_texts(examples):
        result = examples
        result["labels"] = examples["input_ids"].copy()
        return result


    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        # batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    lm_dataset = tokenized_datasets.map(single_texts, batched=True, num_proc=4)
    # lm_dataset = tokenized_dataset
    lm_dataset.set_format('torch', columns=['input_ids', 'attention_mask' ,'labels'])
    
    
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
            
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds = decoded_preds.split(f'输出:\n{tokenizer.bos_token}')[1].split(f"{tokenizer.eos_token}")[0]
        print(decoded_preds)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_labels = decoded_labels.split(f'输出:\n{tokenizer.bos_token}')[1].split(f"{tokenizer.eos_token}")[0]
        print(decoded_labels)
        
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)
        
    
    predict_dataset = lm_dataset["test"]
    
    print(predict_dataset[0])

        
        
    
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='model/chinese-llama-alpaca-plus-lora-7b',
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
    )
    args = parser.parse_args()
    main(args)