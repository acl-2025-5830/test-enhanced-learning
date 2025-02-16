from dataclasses import dataclass, field
from typing import Optional, Dict, List
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    tokenizer_name_or_path: Optional[str] = field(default="")
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    torch_dtype: Optional[str] = field(default="auto")

@dataclass
class DataTrainingArguments:
    method: Optional[str] = field(default=None)
    domain: Optional[str] = field(default=None)
    ratio: Optional[str] = field(default=None)

@dataclass
class MyTrainingArguments(TrainingArguments):
    save_only_model: Optional[bool] = field(default=True)
    do_train : Optional[bool] = field(default=True)
    do_eval : Optional[bool] = field(default=False)
    bf16 : Optional[bool] = field(default=True)
    learning_rate : Optional[float] = field(default=5e-5)
    weight_decay : Optional[float] = field(default=0.01)
    num_train_epochs : Optional[int] = field(default=1)
    eval_strategy : Optional[str] = field(default="no")
    eval_steps : Optional[int] = field(default=1000)
    logging_strategy: Optional[str] = field(default="steps")
    logging_steps: Optional[int] = field(default=1)
    logging_first_step: Optional[bool] = field(default=True)
    save_strategy: Optional[str] = field(default="steps")
    save_steps: Optional[int] = field(default=1000) ### 
    save_total_limit: Optional[int] = field(default=5)
    per_device_train_batch_size: Optional[int] = field(default=1) ###
    gradient_accumulation_steps: Optional[int] = field(default=16)
    warmup_ratio: Optional[float] = field(default=0.03)    
    ddp_find_unused_parameters: Optional[bool] = field(default=True)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    overwrite_output_dir: Optional[bool] = field(default=True)
    remove_unused_columns: Optional[bool] = field(default=True)
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="adamw_hf")