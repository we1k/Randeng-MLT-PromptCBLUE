import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import datasets


TASK_TO_INSTRUCTION = {
    "CHIP-CDEE" : "从下列输入中进行临床发现事件抽取任务。输出临床发现事件的主体词，以及发生状态，描述词和解剖部位这三种属性，其中描述词和解剖部位可能有多个值",
    "CHIP-CDN" : "诊断实体的语义标准化, 从给定的实体选项中选择与原诊断描述匹配的诊断标准词。从实体选项候选输出结果" ,
    "CHIP-CTC" : "根据输入的句子，确定该句子描述的临床试验筛选标准所属的类型。从类型选项候选输出结果",
    "CHIP-MDCFNPC" : "阴阳性判断的任务，在对话中，给出了一系列临床发现实体，然后根据每个实体判断其阴性或阳性。实体包括症状、疾病或假设可能发生的疾病，以及其他医学检查结果。根据对话内容，需要判断每个实体是已有症状疾病、未患有症状疾病，或者回答不明确或无实际意义。",
    # reconstruct
    "CHIP-MDCFNPC" : "阴阳性判断的任务，在对话中，给出了一系列临床发现实体，然后根据每个实体判断其阴性或阳性。实体包括症状、疾病或假设可能发生的疾病，以及其他医学检查结果。根据对话内容，需要判断每个实体是已有症状疾病、未患有症状疾病，或者回答不明确或无实际意义。",
    
    "CHIP-STS": "判断输入中的两句话的意思是否相同。如果两句话意思相同输出\"是的\",意思不相同输出\"不是\"",
    "CMeEE-V2" : "抽取出输入中的医学相关命令实体，并根据提供的选项选择特定类型的实体列表。",
    "CMeIE" :  "从给定的文本中找出特定类型的关系，并找出关系的头实体和尾实体。对每个特定关系三元组输出格式，具有**关系的头尾实体对如下：头实体为**，尾实体为**。如果没有找到实体对。输出\"没有指定类型的三元组\". " ,
    "IMCS-V2-DAC" : "判断输入中给定的问诊句子或陈述句的意图类型。根据所提供的选项，选择输出与句子意图相匹配的答案。", 
    "IMCS-V2-MRG" : "根据下输入中给定的问诊对话生成诊疗报告。输出报告需要包括主诉，现病史，辅助检查，既往史，诊断，建议的内容",
    "IMCS-V2-NER" : "根据给定的输入文本，输出对应的实体类型和实体名称。如果没有找到实体对。输出\"上述句子没有指定类型实体\"",
    "IMCS-V2-SR" : "根据给定的对话历史和当前对话，输出每个对话中涉及的症状以及这些症状的阴阳性判断。如果患有该症状输出\"阳性\",没有患有该症状输出\"阴性\",无法根据上下文确定病人是否患有该症状输出\"无法确定\"",
    # 为什么是相关和不相关呢？ ok or not ok? 能不能换成其他的
    "KUAKE-IR" : "判断输入中的医疗搜索和回答内容是否相关。如果内容相关输出\"相关\",内容不相关输出\"不相关\"",
    "KUAKE-QIC" : "根据输入中的搜索内容句子，判断搜索的意图类型, 从类型选项候选输出结果",
    "KUAKE-QQR" : "判断输入两个句子之间的语义包含关系。是\"完全一致\"，\"后者是前者的语义子集\"，\"后者是前者的语义父集\"，\"语义无直接关联\"的哪一种",
    "KUAKE-QTR" : "判断输入两个句子之间的语义相似程度。是\"完全不匹配或者没有参考价值\"，\"很少匹配有一些参考价值\"，\"部分匹配\"，\"完全匹配\"中的哪一种" ,
    "MedDG" : "根据输入中给定的问诊对话历史生成医生的下一句回复"
}
TASK_TO_MAX_NEW_TOKENS = {
    "CHIP-CDEE" : 256,
    "CHIP-CDN" :  128,
    "CHIP-CTC" : 10,
    "CHIP-MDCFNPC" : 512,
    "CHIP-STS": 4,
    "CMeEE-V2" : 150,
    "CMeIE" :  256,
    "IMCS-V2-DAC" : 32, 
    "IMCS-V2-MRG" : 150,
    "IMCS-V2-NER" : 64,
    "IMCS-V2-SR" : 64,
    "KUAKE-IR" : 7,
    "KUAKE-QIC" : 10,
    "KUAKE-QQR" : 16,
    "KUAKE-QTR" : 16,
    "MedDG" : 100,
}

TASK_TO_TASK_TYPE = {
    "CHIP-CDEE" : "事件三元组抽取",
    "CHIP-CDN" :  "多项选择",
    "CHIP-CTC" : "文本分类",
    "CHIP-MDCFNPC" : "实体阴阳性分析",
    "CHIP-STS" : "语义匹配",
    "CMeEE-V2" : "实体识别",
    "CMeIE" :  "实体三元组抽取",
    "IMCS-V2-DAC" : "意图识别", 
    "IMCS-V2-MRG" : "生成诊疗报告",
    "IMCS-V2-NER" : "实体识别",
    "IMCS-V2-SR" : "实体识别并文本分类",
    "KUAKE-IR" : "相关性分类",
    "KUAKE-QIC" : "意图分类",
    "KUAKE-QQR" : "自然语言推理",
    "KUAKE-QTR" : "语义匹配",
    "MedDG" : "对话生成",
}