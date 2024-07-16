#!/bin/bash
#SBATCH --job-name=eval_cosmo
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus=8
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=1-00:00:00

set -x -e
source ~/.bashrc
source "/admin/home/anton/miniforge3/etc/profile.d/conda.sh"
source activate cosmolighteval

export HF_HOME="/fsx/anton/cosmo/cache/"
export HF_DATASETS_CACHE="/fsx/anton/cosmo/cache/"

MODELS=(
  "openai-community/gpt2"
  "openai-community/gpt2-medium"
  "openai-community/gpt2-xl"
  "karpathy/gpt2_1558M_final4_hf"
  "EleutherAI/pythia-160m"
  "Qwen/Qwen2-0.5B"
  "HuggingFaceTB/cosmo2-1.7B-1T"
  "HuggingFaceTB/cosmo2-149M-600B-fp32"
  "HuggingFaceTB/cosmo2-362M-600B-fp32"
  "HuggingFaceTB/cosmo2-1.7B-900B"
  "HuggingFaceTB/mixture11-600B"
  "HuggingFaceTB/cosmo2-base-magpie-lr-5e-5"
  "HuggingFaceTB/cosmo-300B-with-decay-instruct-mixture-5-bis"
  "HuggingFaceTB/cosmo2-600B-tokens-base-mixture"
  "microsoft/phi-1_5"
  "microsoft/phi-2"
  "HuggingFaceTB/cosmo-1b"
  "HuggingFaceTB/cosmo2-test-classic"
  "HuggingFaceFW/ablation-model-fineweb-edu"
  "HuggingFaceFW/ablation-model-fineweb-v1"
  "Qwen/Qwen1.5-1.8B"
  "Qwen/Qwen2-1.5B"
  "Qwen/Qwen1.5-0.5B"
  "stabilityai/stablelm-2-1_6b"
  "allenai/OLMo-1B-hf"
  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
  "Qwen/Qwen2-1.5B-Instruct"
  "Qwen/Qwen2-0.5B-Instruct"
  "HuggingFaceFW/ablation-model-refinedweb"
  "HuggingFaceFW/ablation-model-c4"
  "HuggingFaceFW/ablation-model-dolma-v1_6"
  "HuggingFaceFW/ablation-model-slimpajama"
  "HuggingFaceFW/ablation-model-the-pile"
  "HuggingFaceFW/ablation-model-redpajama2"
)
OUTPUT_DIR="/fsx/anton/cosmopedia/eval_results_cosmo2/"
OUTPUT_DATASET="HuggingFaceTB/eval_results_cosmo2"

for model in "${MODELS[@]}"
do
  accelerate launch --num_processes=8 --main_process_port=29600 "/admin/home/anton/repos/lighteval/run_evals_accelerate.py" --model_args="pretrained=$model" \
      --custom_tasks "lighteval_tasks.py" --output_dir $OUTPUT_DIR --override_batch_size 15 \
      --tasks "custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|arc:easy|0|1,custom|arc:challenge|0|1,custom|commonsense_qa|0|1,custom|boolq|0|1,custom|trivia_qa|0|1,custom|trivia_qa|5|1,custom|mmlu_pro_cloze|0|1,custom|mmlu_stem_mc|0|1,custom|mmlu_stem_cloze|0|1,custom|gsm8k|5|1,custom|mmlu_mc:abstract_algebra|0|1,custom|mmlu_mc:anatomy|0|1,custom|mmlu_mc:astronomy|0|1,custom|mmlu_mc:business_ethics|0|1,custom|mmlu_mc:clinical_knowledge|0|1,custom|mmlu_mc:college_biology|0|1,custom|mmlu_mc:college_chemistry|0|1,custom|mmlu_mc:college_computer_science|0|1,custom|mmlu_mc:college_mathematics|0|1,custom|mmlu_mc:college_medicine|0|1,custom|mmlu_mc:college_physics|0|1,custom|mmlu_mc:computer_security|0|1,custom|mmlu_mc:conceptual_physics|0|1,custom|mmlu_mc:econometrics|0|1,custom|mmlu_mc:electrical_engineering|0|1,custom|mmlu_mc:elementary_mathematics|0|1,custom|mmlu_mc:formal_logic|0|1,custom|mmlu_mc:global_facts|0|1,custom|mmlu_mc:high_school_biology|0|1,custom|mmlu_mc:high_school_chemistry|0|1,custom|mmlu_mc:high_school_computer_science|0|1,custom|mmlu_mc:high_school_european_history|0|1,custom|mmlu_mc:high_school_geography|0|1,custom|mmlu_mc:high_school_government_and_politics|0|1,custom|mmlu_mc:high_school_macroeconomics|0|1,custom|mmlu_mc:high_school_mathematics|0|1,custom|mmlu_mc:high_school_microeconomics|0|1,custom|mmlu_mc:high_school_physics|0|1,custom|mmlu_mc:high_school_psychology|0|1,custom|mmlu_mc:high_school_statistics|0|1,custom|mmlu_mc:high_school_us_history|0|1,custom|mmlu_mc:high_school_world_history|0|1,custom|mmlu_mc:human_aging|0|1,custom|mmlu_mc:human_sexuality|0|1,custom|mmlu_mc:international_law|0|1,custom|mmlu_mc:jurisprudence|0|1,custom|mmlu_mc:logical_fallacies|0|1,custom|mmlu_mc:machine_learning|0|1,custom|mmlu_mc:management|0|1,custom|mmlu_mc:marketing|0|1,custom|mmlu_mc:medical_genetics|0|1,custom|mmlu_mc:miscellaneous|0|1,custom|mmlu_mc:moral_disputes|0|1,custom|mmlu_mc:moral_scenarios|0|1,custom|mmlu_mc:nutrition|0|1,custom|mmlu_mc:philosophy|0|1,custom|mmlu_mc:prehistory|0|1,custom|mmlu_mc:professional_accounting|0|1,custom|mmlu_mc:professional_law|0|1,custom|mmlu_mc:professional_medicine|0|1,custom|mmlu_mc:professional_psychology|0|1,custom|mmlu_mc:public_relations|0|1,custom|mmlu_mc:security_studies|0|1,custom|mmlu_mc:sociology|0|1,custom|mmlu_mc:us_foreign_policy|0|1,custom|mmlu_mc:virology|0|1,custom|mmlu_mc:world_religions|0|1,custom|mmlu_cloze:abstract_algebra|0|1,custom|mmlu_cloze:anatomy|0|1,custom|mmlu_cloze:astronomy|0|1,custom|mmlu_cloze:business_ethics|0|1,custom|mmlu_cloze:clinical_knowledge|0|1,custom|mmlu_cloze:college_biology|0|1,custom|mmlu_cloze:college_chemistry|0|1,custom|mmlu_cloze:college_computer_science|0|1,custom|mmlu_cloze:college_mathematics|0|1,custom|mmlu_cloze:college_medicine|0|1,custom|mmlu_cloze:college_physics|0|1,custom|mmlu_cloze:computer_security|0|1,custom|mmlu_cloze:conceptual_physics|0|1,custom|mmlu_cloze:econometrics|0|1,custom|mmlu_cloze:electrical_engineering|0|1,custom|mmlu_cloze:elementary_mathematics|0|1,custom|mmlu_cloze:formal_logic|0|1,custom|mmlu_cloze:global_facts|0|1,custom|mmlu_cloze:high_school_biology|0|1,custom|mmlu_cloze:high_school_chemistry|0|1,custom|mmlu_cloze:high_school_computer_science|0|1,custom|mmlu_cloze:high_school_european_history|0|1,custom|mmlu_cloze:high_school_geography|0|1,custom|mmlu_cloze:high_school_government_and_politics|0|1,custom|mmlu_cloze:high_school_macroeconomics|0|1,custom|mmlu_cloze:high_school_mathematics|0|1,custom|mmlu_cloze:high_school_microeconomics|0|1,custom|mmlu_cloze:high_school_physics|0|1,custom|mmlu_cloze:high_school_psychology|0|1,custom|mmlu_cloze:high_school_statistics|0|1,custom|mmlu_cloze:high_school_us_history|0|1,custom|mmlu_cloze:high_school_world_history|0|1,custom|mmlu_cloze:human_aging|0|1,custom|mmlu_cloze:human_sexuality|0|1,custom|mmlu_cloze:international_law|0|1,custom|mmlu_cloze:jurisprudence|0|1,custom|mmlu_cloze:logical_fallacies|0|1,custom|mmlu_cloze:machine_learning|0|1,custom|mmlu_cloze:management|0|1,custom|mmlu_cloze:marketing|0|1,custom|mmlu_cloze:medical_genetics|0|1,custom|mmlu_cloze:miscellaneous|0|1,custom|mmlu_cloze:moral_disputes|0|1,custom|mmlu_cloze:moral_scenarios|0|1,custom|mmlu_cloze:nutrition|0|1,custom|mmlu_cloze:philosophy|0|1,custom|mmlu_cloze:prehistory|0|1,custom|mmlu_cloze:professional_accounting|0|1,custom|mmlu_cloze:professional_law|0|1,custom|mmlu_cloze:professional_medicine|0|1,custom|mmlu_cloze:professional_psychology|0|1,custom|mmlu_cloze:public_relations|0|1,custom|mmlu_cloze:security_studies|0|1,custom|mmlu_cloze:sociology|0|1,custom|mmlu_cloze:us_foreign_policy|0|1,custom|mmlu_cloze:virology|0|1,custom|mmlu_cloze:world_religions|0|1"
done

huggingface-cli upload $OUTPUT_DATASET $OUTPUT_DIR / --repo-type dataset --delete="*"
# huggingface-cli upload HuggingFaceTB/eval_results_cosmo2 /fsx/anton/cosmopedia/eval_results_cosmo2/ / --repo-type dataset --delete="*"