import os
import sys
import json
import csv

path = './aug_data'
samples = []
for file in os.listdir(path):
    new_path = os.path.join(path, file)
    with open(new_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)

output_path = './datasets/PromptCBLUE/aug_train.json'
with open(output_path, 'w', encoding='utf-8' ) as f:
    for sample in samples:
        str = json.dumps(sample, ensure_ascii=False)
        f.write(str+'\n')
