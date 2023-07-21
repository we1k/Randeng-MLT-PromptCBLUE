import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Union, List
import datasets
import torch
import logging
from datasets import load_dataset, concatenate_datasets
import copy
import transformers
import random

from randen_T5.instruction import TASK_TO_INSTRUCTION, TASK_TO_MAX_NEW_TOKENS, TASK_TO_TASK_TYPE

IGNORE_INDEX = -100

logger = logging.getLogger('__name__')


def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_source_length: int, max_target_length: int, ignore_pad_token_for_loss=True, data_cache_dir = None,
                preprocessing_num_workers = None,
                ):

    def tokenization(examples):
        sources = []
        targets = []
        for task, input, output in zip(examples['task_dataset'],examples['input'],examples['target']):
            if input is not None and input !="":
                instruction = TASK_TO_TASK_TYPE[task]+'任务：'+input
            # source = prompt.format_map({'instruction':instruction})
            source = instruction
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        
        model_inputs = tokenizer(sources, max_length=max_source_length, padding=True, truncation=True)
        
        # labels 
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=True, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:

        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0])
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            column_names = raw_dataset['train'].column_names
            print(column_names)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=column_names,
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets



def build_test_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_source_length: int, max_target_length: int, ignore_pad_token_for_loss=True, data_cache_dir = None,
                preprocessing_num_workers = None,
                ):

    def tokenization(examples):
        sources = []
        targets = []
        for input, output in zip(examples['input'],examples['target']):
            if input is not None and input !="":
                instruction = input
            # source = prompt.format_map({'instruction':instruction})
            source = instruction
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        
        model_inputs = tokenizer(sources, max_length=max_source_length, padding=True, truncation=True)
        
        # labels 
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=True, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    logging.warning("building predict dataset...")
    all_datasets = []

    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:

        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0])
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'eval datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            column_names = raw_dataset['test'].column_names
            print(column_names)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=column_names,
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['predict'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )