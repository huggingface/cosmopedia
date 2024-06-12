#!/bin/bash
#SBATCH --job-name=eval_cosmo
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus=4
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
  "HuggingFaceTB/cosmo-1b"
  "HuggingFaceTB/cosmo2-test-classic"
  "HuggingFaceFW/ablation-model-fineweb-edu"
  "HuggingFaceFW/ablation-model-fineweb-v1"
  "HuggingFaceFW/ablation-model-refinedweb"
  "HuggingFaceFW/ablation-model-c4"
  "HuggingFaceFW/ablation-model-dolma-v1_6"
  "HuggingFaceFW/ablation-model-slimpajama"
  "HuggingFaceFW/ablation-model-the-pile"
  "HuggingFaceFW/ablation-model-redpajama2"
  "microsoft/phi-1_5"
  "microsoft/phi-2"
  "Qwen/Qwen1.5-1.8B"
  "Qwen/Qwen2-1.5B"
  "Qwen/Qwen1.5-0.5B"
  "allenai/OLMo-1B-hf"
  "stabilityai/stablelm-2-1_6b"
  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
  "Qwen/Qwen2-1.5B-Instruct"
  "Qwen/Qwen2-0.5B-Instruct"
)
OUTPUT_DIR="/fsx/anton/cosmopedia/eval_results_cosmo2/"
OUTPUT_DATASET="HuggingFaceTB/eval_results_cosmo2"

for model in "${MODELS[@]}"
do
  accelerate launch --num_processes=4 --main_process_port=29600 "/admin/home/anton/repos/lighteval/run_evals_accelerate.py" --model_args="pretrained=$model" \
      --custom_tasks "lighteval_tasks.py" --output_dir $OUTPUT_DIR --override_batch_size 16 \
      --tasks "custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|arc:easy|0|1,custom|arc:challenge|0|1,custom|commonsense_qa|0|1,custom|gsm8k|5|1,custom|mmlu:abstract_algebra|0|1,custom|mmlu:anatomy|0|1,custom|mmlu:astronomy|0|1,custom|mmlu:business_ethics|0|1,custom|mmlu:clinical_knowledge|0|1,custom|mmlu:college_biology|0|1,custom|mmlu:college_chemistry|0|1,custom|mmlu:college_computer_science|0|1,custom|mmlu:college_mathematics|0|1,custom|mmlu:college_medicine|0|1,custom|mmlu:college_physics|0|1,custom|mmlu:computer_security|0|1,custom|mmlu:conceptual_physics|0|1,custom|mmlu:econometrics|0|1,custom|mmlu:electrical_engineering|0|1,custom|mmlu:elementary_mathematics|0|1,custom|mmlu:formal_logic|0|1,custom|mmlu:global_facts|0|1,custom|mmlu:high_school_biology|0|1,custom|mmlu:high_school_chemistry|0|1,custom|mmlu:high_school_computer_science|0|1,custom|mmlu:high_school_european_history|0|1,custom|mmlu:high_school_geography|0|1,custom|mmlu:high_school_government_and_politics|0|1,custom|mmlu:high_school_macroeconomics|0|1,custom|mmlu:high_school_mathematics|0|1,custom|mmlu:high_school_microeconomics|0|1,custom|mmlu:high_school_physics|0|1,custom|mmlu:high_school_psychology|0|1,custom|mmlu:high_school_statistics|0|1,custom|mmlu:high_school_us_history|0|1,custom|mmlu:high_school_world_history|0|1,custom|mmlu:human_aging|0|1,custom|mmlu:human_sexuality|0|1,custom|mmlu:international_law|0|1,custom|mmlu:jurisprudence|0|1,custom|mmlu:logical_fallacies|0|1,custom|mmlu:machine_learning|0|1,custom|mmlu:management|0|1,custom|mmlu:marketing|0|1,custom|mmlu:medical_genetics|0|1,custom|mmlu:miscellaneous|0|1,custom|mmlu:moral_disputes|0|1,custom|mmlu:moral_scenarios|0|1,custom|mmlu:nutrition|0|1,custom|mmlu:philosophy|0|1,custom|mmlu:prehistory|0|1,custom|mmlu:professional_accounting|0|1,custom|mmlu:professional_law|0|1,custom|mmlu:professional_medicine|0|1,custom|mmlu:professional_psychology|0|1,custom|mmlu:public_relations|0|1,custom|mmlu:security_studies|0|1,custom|mmlu:sociology|0|1,custom|mmlu:us_foreign_policy|0|1,custom|mmlu:virology|0|1,custom|mmlu:world_religions|0|1"
done

huggingface-cli upload $OUTPUT_DATASET $OUTPUT_DIR / --repo-type dataset --delete="*"
# huggingface-cli upload HuggingFaceTB/eval_results_cosmo2 /fsx/anton/cosmopedia/eval_results_cosmo2/ / --repo-type dataset --delete="*"