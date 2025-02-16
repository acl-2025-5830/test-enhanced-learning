import os
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Trainer,
    HfArgumentParser,
)

from test_enhanced_learning.arguments import (
    ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments,
)

from test_enhanced_learning.data_utils import (
    DataCollatorForTEL,
    build_TEL,
    build_cpt,
    build_it,
    build_INST_PT,
)
import torch
from test_enhanced_learning.utils import rank0_print, save_args

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
save_args(model_args, data_args, training_args)

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name_or_path,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

build_fn = {
    "CPT": build_cpt,
    "IT": build_it,
    "CPT-IT": build_it,
    "TEL": build_TEL,
    "InstPT": build_INST_PT,
}

train_dataset = build_fn[data_args.method](
    tokenizer=tokenizer,
    domain=data_args.domain,
    method=data_args.method,
)

data_collator = DataCollatorForTEL(tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map=int(os.environ.get("LOCAL_RANK")),
    pad_token_id=tokenizer.pad_token_id,
    trust_remote_code=True,
)


trainable_param_count = 0
rank0_print("="*40)
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_param_count += param.numel()

rank0_print(f"Tranable params : {trainable_param_count/(10**6)} M")
rank0_print("="*40)

training_args.gradient_checkpointing_kwargs={'use_reentrant':False}
training_args.torch_compile=True

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model()
trainer.save_state()