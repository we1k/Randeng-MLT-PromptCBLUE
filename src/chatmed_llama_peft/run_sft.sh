export CUDA_VISIBLE_DEVICES='0,1,2,3'
# export WANDB_MODE=disabled
WANDB_PROJECT=llama_peft

gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo $gpu_num $CUDA_VISIBLE_DEVICES

########参数部分########
lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
# modules_to_save="embed_tokens,lm_head"
modules_to_save="null"
lora_dropout=0.1

pretrained_model=model/chinese-llama-alpaca-plus-lora-7b
dataset_dir=datasets/alpaca
checkpoint_dir=checkpoint

# training setting
per_device_train_batch_size=4
per_device_eval_batch_size=4
training_steps=100
gradient_accumulation_steps=8

deepspeed_config_file=src/chatmed_llama_peft/deepspeed_config_zero3_offload.json

########启动命令########
torchrun --nnodes 1 --nproc_per_node $gpu_num \
    --master_port 29500 \
    src/chatmed_llama_peft/run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --train_file $dataset_dir/train.json \
    --validation_file $dataset_dir/dev.json \
    --fp16 \
    --run_name MLT_lora_alpaca-7b \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --logging_strategy steps \
    --max_steps 3000 \
    --logging_steps 100 \
    --save_steps 200 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 1024 \
    --output_dir $checkpoint_dir/alpaca-lora-$LR \
    --overwrite_output_dir \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_split_percentage 0.001 \
    # --gradient_checkpointing \
    # --ddp_find_unused_parameters False
