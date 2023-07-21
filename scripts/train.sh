# build_instruction_dataset work with usual format 
export CUDA_VISIBLE_DEVICES='2, 3'
export WANDB_LOG_MODEL=true
# export WANDB_MODE=disabled
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo $gpu_num $CUDA_VISIBLE_DEVICES

your_data_path="datasets/PromptCBLUE"  # dataset folder
your_checkpoint_path=checkpoint/randen/new-verb-checkpoint-6000  # checkpoint folder
output_path="checkpoint/randeng-aug-verb"  # output folder
model_path=IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese

# config for deepspeed
deepspeed_config_file="src/chatmed_llama_peft/deepspeed_stage2.json"

# experiment setting
LR=1e-6

torchrun \
    --nnodes 1 \
    --nproc_per_node $gpu_num \
    --master_port 29500 \
    randen_T5/main.py \
    --do_train \
    --train_file $your_data_path/aug_train_verb.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_path \
    --checkpoint_path $your_checkpoint_path \
    --output_dir $output_path \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_steps 10000 \
    --logging_steps 100 \
    --save_steps 1000 \
    --lr_scheduler_type constant \
    --save_total_limit 3 \
    --learning_rate $LR \
    --run_name real_final_verb_15k \
    --report_to wandb
    # --lr_scheduler_type constant \
    # --do_eval \
    # --validation_file $your_data_path/dev.json \
    # --evaluation_strategy steps \
    # --eval_steps 500 \

    # --fp16 \
    # --deepspeed ${deepspeed_config_file} \