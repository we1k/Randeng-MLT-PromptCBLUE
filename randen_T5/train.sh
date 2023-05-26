# build_instruction_dataset work with usual format 
export CUDA_VISIBLE_DEVICES='0,1,2,3'
# export WANDB_MODE=disabled
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo $gpu_num $CUDA_VISIBLE_DEVICES

export WANDB_MODE=disabled
your_data_path="datasets/PromptCBLUE"  # 填入数据集所在的文件夹路径
your_checkpoint_path="checkpoint/randen"  # 填入用来存储模型的路径
model_path=IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese
deepspeed_config_file="src/chatmed_llama_peft/deepspeed_config_zero3_offload.json"

# experiment setting
LR=1e-4


torchrun \
    --nnodes 1 \
    --nproc_per_node $gpu_num \
    --master_port 29500 \
    randen_T5/main.py \
    --deepspeed ${deepspeed_config_file} \
    --fp16 \
    --do_train \
    --train_file $your_data_path/train_dev.json \
    --validation_file $your_data_path/dev.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_path \
    --output_dir $your_checkpoint_path-aug \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_steps 4000 \
    --logging_steps 200 \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate $LR \
    --predict_with_generate true