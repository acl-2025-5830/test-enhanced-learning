# TELLME: Test-Enhanced Learning for Language Model Enrichment



You can find the trained models and datasets (TEL, InstPT) on our anonymous Hugging Face repository. [link](https://huggingface.co/anonymous4459)

## 0. git clone
```bash
git clone https://github.com/acl-2025-5830/test-enhanced-learning.git
cd test-enhanced-learning
```

## 1. Training



### 1.1 Environmental Settings

```bash
python3 -m venv .my_env
source .my_env/bin/activate
pip install -e .
```

### 1.2 Export huggingface token for load models
```
export HF_TOKEN=YOUR_TOKEN
```

### 1.3 Training TEL method
```bash
cd scripts
bash pretrain_tel.sh
```

This script file contains following contents:
```bash
#!/bin/bash
scripts=$(readlink -f "$0")
scripts_dir=$(dirname "$scripts")
base_dir=$(dirname "$scripts_dir")

model_name_or_path="HuggingFaceTB/SmolLM2-1.7B"
tokenizer_name_or_path="HuggingFaceTB/SmolLM2-1.7B"
model_alias="SmolLM2-1.7B"

DOMAIN="finance"
METHOD="TEL"

CUDA_VISIBLE_DEVICES="0,1" torchrun --nnodes 1 --nproc_per_node 2 $base_dir/src/test_enhanced_learning/train/pretrain.py \
    --output_dir $base_dir/saved_models_$DOMAIN/$model_alias-$METHOD\
    --domain $DOMAIN\
    --method $METHOD\
    --save_strategy steps\
    --num_train_epochs 1\
    --model_name_or_path $model_name_or_path\
    --tokenizer_name_or_path $tokenizer_name_or_path\
    --optim adamw_bnb_8bit
```

**Modifying Domains and Methods**
- `DOMAIN` can be switched to either finance or medicine (or extended to other domains of interest if available in your dataset).

- `METHOD` supports different training approaches, including:
    - CPT: Continual Pre-Training
    - IT: Instruction Tuning
    - TEL: Test-Enhanced Learning (default in the snippet)
    - InstPT: Instruction-Based Pre-Training

These options provide flexibility in experimenting with different task objectives and domains, enabling you to compare performance across various scenarios. Simply edit the script to point to the desired DOMAIN and METHOD before running the above command.




## 2. evaluation

### 2.1 enviroment setup
```bash
cd evaluation
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
cd ..
```


### 2.2 Add new tasks
on `evaluation` directory,
```bash

cp -r eval_tasks/ lm-evaluation-harness/lm_eval/tasks/
```
### 2.3 Evaluate on the finance dataset
on `evaluation` directory,

```bash
bash eval_finance.sh
```


### 2.3 Evaluate on the medical dataset
on `evaluation` directory,

```bash
bash eval_medicine.sh
```
