# This repo is the code implement of TianChi Competition: CCKS2023-PromptCBLUE (Ranked 3 / 381 )


# Overview
On based of Chinese MLT Pretrained model `Randeng-MLT`, we finetune it on a Chinese Multitask medical Seq2Seq dataset, PromptCBLUE. 
Also added verbaliser for better and faster model convergence.

# Getting Started:
1. Collect PromptCBLUE data under `aug_data` folder, then run
```python
python data_utils/verbaliser.py
```
2. Start training!
```python
bash scripts/train.sh
```

# Acknowledgements:
Pretrained model: IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese [https://huggingface.co/IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese]
