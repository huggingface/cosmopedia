#!/bin/bash
#SBATCH --job-name=run_edu_bert
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus=1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=1-00:00:00

set -x -e
source ~/.bashrc
source "/admin/home/anton/miniforge3/etc/profile.d/conda.sh"
source activate lighteval

python run_edu_bert.py \
    --model_name="HuggingFaceTB/snowflake_m_edu_reg" \
    --dataset_name="HuggingFaceTB/bisac_3B_generations_college_students" \
    --output_dataset_name="HuggingFaceTB/bisac_3B_generations_college_students_edu_scores" \
    --text_column="text"