import logging
import os
import sys
import json

import numpy as np
from tqdm import tqdm

import datasets
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from torch.utils.data import DataLoader

from argparse import ArgumentParser

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    default_data_collator,
    set_seed,
)

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["WANDB_MODE"]='disabled'

import sys
sys.path.append("./")


from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from randeng.arguments import ModelArguments, DataTrainingArguments
from randeng.instruction import TASK_TO_MAX_NEW_TOKENS, TASK_TO_INSTRUCTION, TASK_TO_TASK_TYPE
from randeng.build_dataset import build_instruction_dataset


from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__)


def main(args):
    
    accelerator = Accelerator()
    device = accelerator.device
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        
    
    # Load pretrained model and tokenizer
    config = T5Config.from_pretrained(
        args.model_name_or_path,
        # trust_remote_code=True
    )

    tokenizer = T5Tokenizer.from_pretrained(
        args.model_name_or_path,
        # trust_remote_code=True
    )
    
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        config=config,
    ).cuda()

    
    # load dataset
    data_path=f"data/{args.task}"
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

    
    # Temporarily set max_target_length for training.
    max_source_length=512
    max_target_length=10
    
    predict_dataset = build_instruction_dataset(
        data_path=[validation_file],
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        ignore_pad_token_for_loss=True,
        preprocessing_num_workers=8
    )


    data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    predict_dataloader = DataLoader(
        predict_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    
    # prepare everything with accelerator
    model, predict_dataloader = accelerator.prepare(
        model, predict_dataloader
    )
    
    model.eval()
    # 
    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes
    logger.info("***** Running predictions *****")
    logger.info(f"  Num examples = {len(predict_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    
    # 读取原test file
    list_test_samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        line = f.readline()
        list_test_samples = json.loads(line)

    all_pred = []
    for step, batch in enumerate(predict_dataloader):
        inputs = {k:v.to(device) for k, v in batch.items()}
        inputs['max_new_tokens'] = TASK_TO_MAX_NEW_TOKENS[args.task]
        output = model.generate(**inputs)
        predictions = tokenizer.batch_decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        # inputs["labels"] = inputs["labels"].where(inputs["labels"]==-100, tokenizer.pad_token_id)
        # labels = tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        # print(predictions, labels)
        predictions = [pred.strip() for pred in predictions]
        all_pred += predictions
        
                
    output_prediction_file = os.path.join(args.output_dir, f'/test_prediction.json')
        
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        for idx, p in enumerate(all_pred):
            samp = list_test_samples[idx]
            samp["output"] = p
            res = json.dumps(samp, ensure_ascii=False)
            writer.write(f"{res}\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/CHIP-STS',
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese',
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='alpaca'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=32
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()
    main(args)