# This script is based on the modification from https://github.com/huggingface/transformers
import logging
import os
import random
import sys
import json

import torch
from datasets import DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

from arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

max_source_length = 512
max_train_samples = None

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True,truncation_side="left")

set_seed(2023)
with open('medi-data/medi-data.json') as f:
    train_examples_raw = json.load(f)

old_train_examples_raw = train_examples_raw
train_examples_raw = []
total_n = len(old_train_examples_raw)
real_batch_size = max(training_args.per_device_train_batch_size,
                      training_args.per_device_train_batch_size * torch.cuda.device_count())
# print('real_batch_size: ', real_batch_size,training_args.per_device_train_batch_size,torch.cuda.device_count())
for idx in range(0, total_n, real_batch_size):
    local_task_name = old_train_examples_raw[idx]['task_name']
    cur_batch = []
    include_batch = True
    for idx1 in range(idx, min(idx + real_batch_size, total_n)):
        if not old_train_examples_raw[idx1]['task_name'] == local_task_name:
            print(f'one batch in task {old_train_examples_raw[idx1]["task_name"]} is skipped')
            include_batch = False
            break
        else:
            cur_batch.append(old_train_examples_raw[idx1])
    if include_batch and len(cur_batch) == real_batch_size:
        train_examples_raw.append(cur_batch)
random.shuffle(train_examples_raw)
train_examples_raw_batch = train_examples_raw
train_examples_raw = []
for b in train_examples_raw_batch:
    train_examples_raw += b
print(f'There are {len(train_examples_raw)} pairs to train in total')

train_examples = {'query': [], 'pos': [], 'neg': [], 'task_name': []}
task_name_map = {}
total_train_num = len(train_examples_raw)
task_count = 0
for i in range(total_train_num):
    cur_e = train_examples_raw[i]
    for k in ['query', 'pos', 'neg']:
        for s in cur_e[k][:-1]:
            assert not '!@#$%^&**!@#$%^&**' in s
        cur_e[k][-1] = str(cur_e[k][-1])
        if True: # not data_args.add_prompt_to_document:
            cur_e[k][0] = ''
        assert cur_e[k][0].startswith('Represent ') or cur_e[k][0] == ''
        train_examples[k].append('!@#$%^&**!@#$%^&**'.join(cur_e[k]))  # '<prompt>!@#$%^&**!@#$%^&**<document>'
    if not cur_e['task_name'] in task_name_map:
        task_name_map[cur_e['task_name']] = task_count
        task_count += 1
    train_examples['task_name'].append(task_name_map[cur_e['task_name']])

raw_datasets = DatasetDict({'train': Dataset.from_dict(train_examples)})

column_names = raw_datasets["train"].column_names


def preprocess_function(examples):
    all_tokenized = None
    for key in ['query', 'pos', 'neg']:
        num = len(examples[key])
        contexts = []
        for local_idx in range(num):
            splits = examples[key][local_idx].split('!@#$%^&**!@#$%^&**')
            # assert len(splits) == 2
            contexts.append(splits[-1])
            assert isinstance(contexts[-1], str)
        tokenized = tokenizer([q + '[MASK]' for q in contexts], padding=True, truncation=True,
                              return_tensors="pt", max_length=max_source_length)
        # tokenized['context_masks'] = torch.sum(context_tok['attention_mask'], dim=1)
        # tokenized['context_masks'] = tokenized['context_masks'] - 1
        # for my_idx in range(len(tokenized['context_masks'])):
        #     if tokenized['context_masks'][my_idx] <= 1:
        #         tokenized['context_masks'][my_idx] = 0
        keys = tokenized.keys()
        if all_tokenized is None:
            all_tokenized = tokenized.copy()
            for k in keys:
                all_tokenized[k] = all_tokenized[k].tolist()
        for k in keys:
            all_tokenized[f'{key}_{k}'] = tokenized[k].tolist()
    all_tokenized['task_name'] = examples['task_name']
    # all_tokenized['label'] = all_tokenized['input_ids']
    return all_tokenized


train_dataset = raw_datasets["train"]
if max_train_samples is not None:
    max_train_samples = min(len(train_dataset), max_train_samples)
    train_dataset = train_dataset.select(range(max_train_samples))
# with TrainingArguments(output_dir='/medi-data').main_process_first(desc="train dataset map pre-processing"):
with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = train_dataset.map(
        preprocess_function,
        # batch_size=1,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )
train_dataset.save_to_disk('medi-data/processed')