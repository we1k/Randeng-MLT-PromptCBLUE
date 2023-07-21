# build_instruction_dataset work with usual format
export CUDA_VISIBLE_DEVICES='0,2,3'
# export WANDB_LOG_MODEL=true
export WANDB_MODE=disabled
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo $gpu_num $CUDA_VISIBLE_DEVICES

your_data_path="datasets/PromptCBLUE"  # 填入数据集所在的文件夹路径
your_checkpoint_path="checkpoint/randen"  # 填入用来存储模型的路径
checkpoint_name=last-last-verb-checkpoint-14000

output_path=output/test

model_path=IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese

# generation config
generation_config=config/config.json

torchrun \
    --nnodes 1 \
    --nproc_per_node $gpu_num \
    --master_port 29501 \
    src/randeng/main.py \
    --do_predict \
    --checkpoint_path $your_checkpoint_path/$checkpoint_name \
    --test_file $your_data_path/test.json \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_path \
    --output_dir $output_path/$checkpoint_name \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 256 \
    --per_device_eval_batch_size 32 \
    --predict_with_generate
    # --generation_config $generation_config \
