# build_instruction_dataset work with usual format 

export WANDB_MODE=disabled
your_data_path="data/CHIP-STS"  # 填入数据集所在的文件夹路径
your_checkpoint_path="checkpoint"  # 填入用来存储模型的路径
model_path=IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese


# experiment setting
LR=1e-4

CUDA_VISIBLE_DEVICES=1 python randen_T5/main.py \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_path \
    --output_dir T5_toy_dir \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --max_steps 50 \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate $LR \
    --predict_with_generate true
    # --do_train \