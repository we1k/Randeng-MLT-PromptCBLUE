export WANDB_MODE=disabled
your_data_path="datasets/PromptCBLUE"  # 填入数据集所在的文件夹路径
your_checkpoint_path="checkpoint/randen"  # 填入用来存储模型的路径
model_path=IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese

STEP=2000    # 用来评估的模型checkpoint是训练了多少步

CUDA_VISIBLE_DEVICES=0 python randen_T5/main.py \
    --do_predict \
    --checkpoint_path $your_checkpoint_path/checkpoint-$STEP \
    --validation_file $your_data_path/dev.json \
    --test_file $your_data_path/test.json \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_path \
    --output_dir $your_checkpoint_path \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
