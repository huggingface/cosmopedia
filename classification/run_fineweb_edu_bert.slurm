#!/bin/bash
#SBATCH --job-name=fine_edu_bert
#SBATCH --partition hopper-prod
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus=1
#SBATCH -o /fsx/anton/logs/%x_%j.out
#SBATCH -e /fsx/anton/logs/%x_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --array=0-127%128

set -x -e
source ~/.bashrc
source "/admin/home/anton/miniforge3/etc/profile.d/conda.sh"
source activate lighteval

python run_fineweb_edu_bert.py \
    --model_name="HuggingFaceTB/snowflake_m_edu_reg" \
    --input_path="fineweb-data-processing-us-east-1/pii_removed/*/*.jsonl.gz" \
    --output_path="cosmopedia-data/fineweb_edu_scores/" \
    --tmp_path="cosmopedia-data/tmp_fineweb_edu_scores/" \
    --shard ${SLURM_ARRAY_TASK_ID} \
    --num_shards 128