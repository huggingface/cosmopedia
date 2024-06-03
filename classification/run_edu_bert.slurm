#!/bin/bash
#SBATCH --job-name=run_edu_bert
#SBATCH --partition hopper-prod
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus=1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --array=0-127%128

set -x -e
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
source activate pytorch

python run_edu_bert.py \
    --model_name="HuggingFaceFW/fineweb-edu-classifier" \
    --dataset_name="HuggingFaceFW/fineweb" \
    --dataset_config="CC-MAIN-2019-04" \
    --output_dataset_name="HuggingFaceFW/fineweb-edu-annotations" \
    --output_dataset_config="CC-MAIN-2019-04" \
    --text_column="text" \
    --shard ${SLURM_ARRAY_TASK_ID} \
    --num_shards 128
