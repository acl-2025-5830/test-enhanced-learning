# TELLME: Test-Enhanced Learning for Language Model Enrichment
This repository accompanies our paper submission on TELLME (Test-Enhanced Learning for Language Model Enrichment). It provides code and instructions for training, evaluating, and visualizing results for language models using our proposed method.
You can also find the trained models and datasets (TEL, InstPT) on our anonymous Hugging Face repository.  [link](https://huggingface.co/anonymous4459)

# Model & Dataset: https://huggingface.co/anonymous4459

You can find the TEL dataset at [anonymous4459/TEL_train_datasets](https://huggingface.co/datasets/anonymous4459/TEL_train_datasets), and the InstPT dataset at [anonymous4459/InstPT_train_datasets](https://huggingface.co/datasets/anonymous4459/InstPT_train_datasets).



## 0. Clone the Repository
```bash
git clone https://github.com/acl-2025-5830/test-enhanced-learning.git
cd test-enhanced-learning
```

## 1. Training

### 1.1 Environment Setup

```bash
python3 -m venv .my_env
source .my_env/bin/activate
pip install -e .
```

### 1.2 Export huggingface token for load models
```
export HF_TOKEN=YOUR_TOKEN
```

### 1.3 Training with TEL Method
```bash
cd scripts
bash pretrain_tel.sh
```

The `pretrain_tel.sh` script contains the following:
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




## 2. Evaluation

### 2.1 Enviroment setup
```bash
cd evaluation
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
cd ..
```


### 2.2 Adding New Tasks for Evaluation
on `evaluation` directory,
```bash

cp -r eval_tasks/ lm-evaluation-harness/lm_eval/tasks/
```
### 2.3 Evaluate on the finance dataset
on `evaluation` directory,

```bash
bash eval_finance.sh
```


### 2.4 Evaluate on the medical dataset
on `evaluation` directory,

```bash
bash eval_medicine.sh
```


## 3. Visualization

You can find the visualized results and code on `evaluation/visualization`.

Figure 3: Perplexity comparison on the medical dataset for different training approaches. 

![image](https://github.com/user-attachments/assets/2e8499df-d088-4d21-9c36-0dfcc45735b1)



Figure 4: Performance of the finance domain after overwriting with medicine data.

![image](https://github.com/user-attachments/assets/e1558d1e-5dc0-484e-bf1b-c92056a6ac83)


Figure 5: Perplexity scores of CPT and TELLME methods based on training steps.

![image](https://github.com/user-attachments/assets/d1042482-233d-424f-85e9-24c7b17715a1)


**Thank you for using TELLME!**
