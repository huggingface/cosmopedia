#!/bin/bash
#SBATCH --job-name=train_edu_bert
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

python train_edu_bert.py \
    --base_model_name="Snowflake/snowflake-arctic-embed-m" \
    --dataset_name="HuggingFaceTB/LLM_juries_fineweb_430k_annotations" \
    --target_column="median_score"\
    --checkpoint_dir="/fsx/anton/cosmopedia/edu_score/snowflake_regression_median_jury"