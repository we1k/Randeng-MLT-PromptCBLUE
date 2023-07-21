import re
import os
import sys
import json
import csv
import random

path = './aug_data'
samples = []
for file in os.listdir(path):
    new_path = os.path.join(path, file)
    with open(new_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            if sample['task_dataset'] == 'IMCS-V2-DAC': continue
            samples.append(sample)

NULL_keys = ['CHIP-CTC' , 'KUAKE-QIC']
with_first_sent_keys = ["CMeEE-V2", "IMCS-V2-MRG",]
# MN SR CTC

for sample in samples:
    # 添加非上述类型
    # if sample['task_dataset'] in NULL_keys:
    #     prompt = sample['input']
    #     if prompt[-1] == '：':
    #         prompt = prompt[:-3]
    #     prompt += "，非上述类型"
    #     sample['input'] = prompt

    # CMeIE
    if sample['task_dataset'] == 'CMeIE':
        output = ""
        if sample['target'] == '':
            continue
        lines = sample['target'].split('\n')
        # 定义分隔符
        for line in lines:
            if line.startswith('没有'):
                for choice in sample['answer_choices']:
                    output += f"\n{choice}关系："
            else:
                rel_pat = r"具有(.*?)关系的头尾实体"
                rel = re.findall(rel_pat, line)[0].strip()
                output += f"\n{rel}关系："
                pat = r"头实体为(.*?)，尾实体为(.*?)。"
                matches = re.findall(pat, line)
                dict = {}
                for match in matches:
                    sub, obj = match[0].strip(), match[1].strip()
                    if sub in dict:
                        dict[sub].append(obj)
                    else:
                        dict[sub] = [obj]
                for k, v in dict.items():
                    output += f'({k}，[{"|".join(v)}])。'
        sample['target'] = output[1:]

    # 修改CHIP-CDEE NER输出
    elif sample['task_dataset'] == 'CHIP-CDEE':
        output = ""
        lines = sample['target'].split('\n')[1:]
        # 定义分隔符
        delimiters = ["主体词：", "发生状态：", "描述词：", "解剖部位："]
        for line in lines:
            triple = []
            for delimiter in delimiters:
                # 按照分隔符进行切分
                split_string = line.split(delimiter, 1)
                
                # 提取切分后的内容
                if len(split_string) > 1:
                    value = split_string[1].split("；", 1)[0].strip()
                else:
                    value = ""
                triple.append(value)
            output += "\n(" + '；'.join(triple) +")"
        sample['target'] = output[1:]
    
    # 修改MDCFNPC 
    elif sample['task_dataset'] == 'CHIP-MDCFNPC':
        output = ""
        lines = sample['target'].split('\n')[1:] 
        for line in lines:
            # print(line.split('：'))
            sym, label = line.split('：')[0], line.split('：')[1]
            if label.startswith("已有"):
                label = "阳阳阳"
            elif label.startswith("未患有"):
                label = '阴阴阴'
            elif label.startswith("没有回答"):
                label = '其他的'
            elif label.startswith('无实际'):
                label = '不标注'
            output += f"\n{sym}：{label}"
        sample['target'] = output[1:]
    
    # 修改STS
    if sample['task_dataset'] == 'CHIP-STS':
        label = sample['target']
        if label in ['不同', '不是']:
            label = '不'
        else:
            label = '是'
        sample['target'] = ''.join([label for _ in range(10)])
    
    # 修改CTC
    elif sample['task_dataset'] == 'CHIP-CTC':
        label = sample['target']
        sample['target'] = '；'.join([label for _ in range(3)])
    
    
    # 修改QIC
    elif sample['task_dataset'] == 'KUAKE-QIC':
        label = sample['target']
        sample['target'] = '；'.join([label for _ in range(3)])

    
    # 修改KUAKE-IR
    elif sample['task_dataset'] == 'KUAKE-IR':
        label = sample['target']
        sample['target'] = '；'.join([label for _ in range(3)])
    
    # 修改IMCS-SR
    elif sample['task_dataset'] == 'IMCS-V2-SR':
        output = ""
        lines = sample['target'].split('\n')[1:] 
        for line in lines:
            sym, label = line.split('：')[0], line.split('：')[1]
            if label.startswith("患有"):
                label = "阳阳阳"
            elif label.startswith("没有患有"):
                label = '阴阴阴'
            elif label.startswith("无法根据"):
                label = '不确定'
            output += f"\n{sym}：{label}"
        sample['target'] = output[1:]
    
    # # 去掉第一行
    # elif sample['task_dataset'] in with_first_sent_keys:
    #     lines = sample['target'].split('\n')[1:]
    #     sample['target'] = '\n'.join(lines)
    
    # samples.append(sample)

random.shuffle(samples)

## write output 
# output_path = './datasets/toy_examples/toy_aug_train_verb.json'
output_path = './datasets/PromptCBLUE/aug_train_verb.json'
with open(output_path, 'w', encoding='utf-8') as f:
    for sample in samples:
        str = json.dumps(sample, ensure_ascii=False)
        f.write(str+'\n')
