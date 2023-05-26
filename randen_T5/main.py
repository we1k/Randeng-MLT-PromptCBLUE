#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

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

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

import sys
sys.path.append("./")

from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from randen_T5.build_dataset import build_instruction_dataset
from randen_T5.arguments import ModelArguments, DataTrainingArguments



logger = logging.getLogger(__name__)

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config = T5Config.from_pretrained(
        model_args.model_name_or_path,
        # trust_remote_code=True
    )

    tokenizer = T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        # trust_remote_code=True
    )
    
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    ).cuda()

    if model_args.checkpoint_path != None and os.path.exists(model_args.checkpoint_path):
        logger.info(f'loading checkpoint from {model_args.checkpoint_path}')
        weights_file = os.path.join(model_args.checkpoint_path, 'pytorch_model.bin')
        state_dict = torch.load(weights_file, map_location='cpu')
        model.load_state_dict(state_dict)
        del state_dict
        model = model.cuda()
    
    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad)

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    
    # Temporarily set max_target_length for training.
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length



    def print_dataset_example(example):
        print("input_ids",example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        example["labels"] = [l if l != -100 else tokenizer.pad_token_id for l in example['labels']]
        print("labels", tokenizer.decode(example["labels"]))

    if training_args.do_train:
        with training_args.main_process_first(desc="loading and tokenization"):
            train_dataset = build_instruction_dataset(
                data_path=[data_args.train_file],
                tokenizer=tokenizer,
                max_source_length=data_args.max_source_length,
                max_target_length=data_args.max_target_length,
                ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
                preprocessing_num_workers=data_args.preprocessing_num_workers
            )
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        print_dataset_example(train_dataset[0])

    if training_args.do_eval:
        with training_args.main_process_first(desc="loading and tokenization"):
            eval_dataset = build_instruction_dataset(
                data_path=[data_args.validation_file],
                tokenizer=tokenizer,
                max_source_length=data_args.max_source_length,
                max_target_length=data_args.max_target_length,
                ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
                preprocessing_num_workers=data_args.preprocessing_num_workers
            )
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("eval example:")
    
    if training_args.do_predict:
        with training_args.main_process_first(desc="loading and tokenization"):
            predict_dataset = build_instruction_dataset(
                data_path=[data_args.test_file],
                tokenizer=tokenizer,
                max_source_length=data_args.max_source_length,
                max_target_length=data_args.max_target_length,
                ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
                preprocessing_num_workers=data_args.preprocessing_num_workers
            )
        logger.info(f"Num predict_samples  {len(predict_dataset)}")
        logger.info("eval example:")
    
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=True
    )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=512, temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # 读取原test file
        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            # max_tokens=512,
            max_new_tokens=data_args.max_target_length,
            # do_sample=True,
            # top_p=0.7,
            # temperature=0.95,
            # repetition_penalty=1.1
        )
        metrics = predict_results.metrics
        print(metrics)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]

                output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")

                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for idx, p in enumerate(predictions):
                        samp = list_test_samples[idx]
                        samp["target"] = p
                        res = json.dumps(samp, ensure_ascii=False)
                        writer.write(f"{res}\n")

    return results



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()