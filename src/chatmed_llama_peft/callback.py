import os
import shutil
import sys

import wandb

import transformers
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    TrainingArguments,
    TrainerControl,
    TrainerState,
)

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        
        # delete total model state_dict to save disk space
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        
        # delete deepspeed model's state_dict to save disk space
        deepspeed_path = os.path.join(checkpoint_folder, f"global_step{state.global_step}")
        if os.path.exists(deepspeed_path):
            shutil.rmtree(deepspeed_path, ignore_errors=True)
            
        return control