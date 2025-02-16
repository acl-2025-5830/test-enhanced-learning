#!/bin/bash
scripts=$(readlink -f "$0")
scripts_dir=$(dirname "$scripts")
base_dir=$(dirname "$scripts_dir")

model_name_or_path="HuggingFaceTB/SmolLM2-1.7B"
tokenizer_name_or_path="HuggingFaceTB/SmolLM2-1.7B"
model_alias="SmolLM2-1.7B"
echo "######################## Training model: " $model_name_or_path "####################################"

DOMAIN="finance"
METHOD="TEL"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nnodes 1 --nproc_per_node 8 $base_dir/src/test_enhanced_learning/train/pretrain.py \
    --output_dir $base_dir/saved_models_$DOMAIN/$model_alias-$METHOD\
    --domain $DOMAIN\
    --method $METHOD\
    --save_strategy steps\
    --num_train_epochs 1\
    --model_name_or_path $model_name_or_path\
    --tokenizer_name_or_path $tokenizer_name_or_path\
    --optim adamw_bnb_8bit