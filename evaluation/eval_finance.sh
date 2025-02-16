HF_TOKEN="YOUR HUGGINGFACE TOKEN"
OUTPUT_PATH="./"
model_list=(
    'anonymous4459/SmolLM2-1.7B-finance-TEL'
)

N_SHOT=4

for model_path in "${model_list[@]}"
do
    model_dir="$(dirname "${model_path}")"
    model_name="$(basename "${model_dir}")"

    echo "=========================================="
    echo "model_path:  ${model_path}"
    echo "model_dir:   ${model_dir}"
    echo "model_name:  ${model_name}"
    echo "=========================================="

    accelerate launch -m lm_eval \
        --model hf \
        --tasks mmlu_continuation_business_ethics,mmlu_continuation_econometrics,mmlu_continuation_high_school_macroeconomics,mmlu_continuation_high_school_microeconomics,mmlu_continuation_management,mmlu_continuation_marketing,mmlu_continuation_professional_accounting\
        --batch_size 1 \
        --model_args "pretrained=${model_path},dtype=bfloat16,token=${HF_TOKEN}" \
        --output_path "${OUTPUT_PATH}" \
        --num_fewshot ${N_SHOT} \
        --trust_remote_code\
        --seed 42

    accelerate launch -m lm_eval \
        --model hf \
        --tasks nifty,fomc\
        --batch_size 1 \
        --model_args "pretrained=${model_path},dtype=bfloat16,token=${HF_TOKEN}" \
        --output_path "${OUTPUT_PATH}" \
        --trust_remote_code\
        --seed 42
done