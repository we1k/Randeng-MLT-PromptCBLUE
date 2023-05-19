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


from trainer import Trainer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES']='4'
os.environ["WANDB_MODE"]='disabled'


import sys
sys.path.append("./")
from trainer import Trainer

from peft import LoraConfig, get_peft_model, TaskType

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(torch.cuda.get_device_name())
print(f'device_count {torch.cuda.device_count()}')
# model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
# tokenizer = AutoTokenizer.from_pretrained('gpt2')

model_name_or_path = 'model/chinese-llama-alpaca-plus-lora-7b'
config = LlamaConfig.from_pretrained(
    model_name_or_path,
    # trust_remote_code=True
)
tokenizer = LlamaTokenizer.from_pretrained(
    model_name_or_path,
    # trust_remote_code=True
)

model = LlamaForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
).half().cuda()

from peft import PeftConfig, LoraConfig, PeftModelForCausalLM, get_peft_model
checkpoint_name = '../checkpoint/CHIP-CTC-1e-4/checkpoint-300/adapter_model'
peft_config = LoraConfig.from_pretrained(checkpoint_name)
model = get_peft_model(model, peft_config)
model = PeftModelForCausalLM.from_pretrained(model, checkpoint_name)





your_data_path="datasets/toy_examples"
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


# Get the column names for input/target.
prompt_column = 'input'
response_column = 'target'

column_names = lm_datasets["validation"].column_names
# Temporarily set max_target_length for training.
max_target_length = 196
max_input_length = 256
prefix = ''

def generate_prompt(instruction, data):
    return f"""
    ### 指令:\n{instruction}\n### 输入:\n{data[prompt_column]}### 输出:
    {tokenizer.bos_token + data[response_column] + tokenizer.eos_token}
    """
    
def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 1024
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def preprocess_function(data_point):
    # instruction = TASK_TO_INSTRUCTION[data_point['task_type']]
    # instruction = TASK_TO_INSTRUCTION[data_point['task_dataset']]
    full_prompt = generate_prompt("", data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

# Main data processing function that will make each entry its own in the dataset
def single_texts(examples):
    result = examples
    result["labels"] = examples["input_ids"].copy()
    return result


tokenized_datasets = lm_datasets.map(
    preprocess_function,
    # batched=True,
    num_proc=4,
    remove_columns=column_names,
    load_from_cache_file=True,
)

lm_dataset = tokenized_datasets.map(single_texts, batched=True, num_proc=1)
# lm_dataset = tokenized_dataset
lm_dataset.set_format('torch', columns=['input_ids', 'labels'])

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
        
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print(decoded_preds)
    print(labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(decoded_labels)
    
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

import transformers
args = TrainingArguments(
    output_dir='toy_dir',
    run_name='toy_run',
    do_predict=True,
    evaluation_strategy='epoch',
    per_device_eval_batch_size=1,
    save_total_limit=2,
    # report_to='wandb',
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)


class Data_args:
    def __init__(self):
        self.val_max_target_length = 200
        self.eval_beams = 1

print(args.local_rank)
args.local_rank = -1
data_args = Data_args()
args.predict_with_generate = True

trainer = Trainer(
    model,
    args=args,
    data_args=data_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=lm_dataset['train'],
    eval_dataset=lm_dataset['validation'],
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# trainer.evaluate()
test_file = 'datasets/toy_examples/test.json'
list_test_samples = []
with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        list_test_samples.append(line)


predict_result = trainer.predict(lm_dataset['test'])
# trainer.log_metrics('predict', predict_result.metrics)
trainer.save_metrics('predict', predict_result.metrics)
print(predict_result.metrics)

predictions = tokenizer.batch_decode(
    predict_result.predictions, skip_special_tokens=False,
)
pred_file = 'dev/toy_pred.json'

with open(pred_file, "w", encoding="utf-8") as writer:
    for idx, p in enumerate(predictions):
        samp = list_test_samples[idx]
        samp["target"] = p
        res = json.dumps(samp, ensure_ascii=False)
        writer.write(f"{res}\n")