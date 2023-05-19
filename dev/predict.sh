export CUDA_VISIBLE_DEVICES='0,1,2,3'
export WANDB_MODE='disabled'

# export WANDB_MODE=disabled
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo $gpu_num $CUDA_VISIBLE_DEVICES

# experiment setting
lora_rank=8
lora_alpha=32
# lora_trainable="q_proj,k_proj,down_proj,up_proj,gate_proj"
lora_trainable=".*(q_proj|k_proj|down_proj|up_proj|gate_proj)"
modules_to_save="null"
# modules_to_save="embed_tokens,lm_head"
lora_dropout=0.1
LR=1e-4


your_checkpoint_path="checkpoint"  # 填入用来存储模型的路径
your_data_path="data"
deepspeed_config_file="src/chatmed_llama_peft/deepspeed_config_zero3_offload.json"

# task=("CMeEE-V2" "CHIP-CTC" "IMCS-V2-DAC" "CHIP-STS" "IMCS-V2-MRG" "MedDG" "CHIP-CDN" "IMCS-V2-NER" "CHIP-MDCFNPC" "KUAKE-QTR" "KUAKE-QIC" "CMeIE" "IMCS-V2-SR" "KUAKE-QQR" "KUAKE-IR")
tasks=("CHIP-CDEE")

your_checkpoint_path="checkpoint"  # 填入用来存储模型的路径
STEP=300    # 用来评估的模型checkpoint是训练了多少步


for task in "${tasks[@]}"
do
    echo "Running task: " $task

    torchrun \
        --nnodes 1 \
        --nproc_per_node $gpu_num \
        --master_port 29501 \
        src/chatmed_llama_peft/main.py \
        --deepspeed ${deepspeed_config_file} \
        --do_predict \
        --test_file $your_data_path/$task/test.json \
        --prompt_column input \
        --response_column target \
        --model_name_or_path model/chinese-llama-alpaca-plus-lora-7b \
        --peft_path $your_checkpoint_path/$task-$LR/checkpoint-$STEP/adapter_model \
        --output_dir $your_checkpoint_path/$task-$LR \
        --task_name $task \
        --per_device_eval_batch_size 16 \
        --logging_steps 20 \
        --learning_rate $LR \
        --lora_rank ${lora_rank} \
        --lora_alpha ${lora_alpha} \
        --trainable ${lora_trainable} \
        --modules_to_save ${modules_to_save} \
        --lora_dropout ${lora_dropout}
done