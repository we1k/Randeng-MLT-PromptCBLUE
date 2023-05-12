export CUDA_VISIBLE_DEVICES='0,1,2,3,7'
export WANDB_MODE=disabled
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

# experiment setting
lora_rank=8
lora_alpha=32

# lora_trainable="q_proj,k_proj,down_proj,up_proj,gate_proj"
lora_trainalbe=".*(2[0-9]|3[0-1]).*(q_proj|k_proj|down_proj|up_proj|gate_proj)"
modules_to_save="null"
# modules_to_save="embed_tokens,lm_head"
lora_dropout=0.1
LR=2e-2
# your_data_path="datasets/toy_examples"  # 填入数据集所在的文件夹路径
your_data_path="datasets/PromptCBLUE"  # 填入数据集所在的文件夹路径
your_checkpoint_path="checkpoint"  # 填入用来存储模型的路径
deepspeed_config_file="src/chatmed_llama_peft/deepspeed_config_zero3_offload.json"
# peft_path="" 

torchrun \
    --nnodes 1 \
    --nproc_per_node $gpu_num \
    --master_port 29500 \
    src/chatmed_llama_peft/main.py \
    --deepspeed ${deepspeed_config_file} \
    --fp16 \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --test_file $your_data_path/test.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path model/chinese-llama-alpaca-plus-lora-7b \
    --output_dir $your_checkpoint_path/PromptCBLUE-alpaca-llama-7b-lora-$LR \
    --run_name alpaca-llama-7b-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_steps 5000 \
    --logging_steps 20 \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout}
    # --eval_accumulation_step \
    # --report_to wandb \
    # --peft_path $peft_path \
    # --pre_seq_len $PRE_SEQ_LEN \
    # --resume_from_checkpoint checkpoint/toy-PromptCBLUE-alpaca-llama-7b-lora-2e-4/checkpoint-10 \


