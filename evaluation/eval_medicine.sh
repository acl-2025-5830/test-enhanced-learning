HF_TOKEN="YOUR HUGGINGFACE TOKEN"
OUTPUT_PATH="./"
model_list=(
    'anonymous4459/SmolLM2-1.7B-medicine-TEL'
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
        --tasks medmcqa,mmlu_continuation_clinical_knowledge,mmlu_continuation_medical_genetics,mmlu_continuation_anatomy,mmlu_continuation_professional_medicine,mmlu_continuation_college_biology,mmlu_continuation_college_medicine,headqa_en\
        --batch_size 1 \
        --model_args "pretrained=${model_path},dtype=bfloat16,token=${HF_TOKEN}" \
        --output_path "${OUTPUT_PATH}" \
        --trust_remote_code\
        --num_fewshot ${N_SHOT}\
        --seed 42
done