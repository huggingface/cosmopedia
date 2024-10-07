#!/bin/bash
#SBATCH --partition=hopper-prod
#SBATCH --job-name=run_llama
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --array=0-31%4
#SBATCH --time=4-00:00:00

set -x -e
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
source activate pyspark

module load cuda/12.1

python run_llama.py --input_path "s3://cosmopedia-data/re_extract_cc/llama_math/owm_candidates/" \
    --output_path "s3://cosmopedia-data/re_extract_cc/llama_math/owm_llama_3_1_labels/" \
    --shard $SLURM_ARRAY_TASK_ID --num_shards 32