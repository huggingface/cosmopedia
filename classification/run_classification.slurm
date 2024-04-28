#!/bin/bash
#SBATCH --job-name=classification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-cpu
#SBATCH --cpus-per-task=48
#SBATCH --output=/fsx/loubna/logs/embeddings/classification-%x-%n-%j.out
#SBATCH --error=/fsx/loubna/logs/embeddings/classification-%x-%n-%j.err


source ~/.bashrc
source /admin/home/loubna/miniconda3/etc/profile.d/conda.sh
conda activate textbooks 

export HUGGINGFACE_HUB_CACHE=/fsx/loubna/.cache
export HF_DATASETS_CACHE=/fsx/loubna/.cache
export HF_MODULES_CACHE=/fsx/loubna/.cache


set -x -e
echo "START TIME: $(date)"
echo python3 version = `python3 --version`

start=$(date +%s)
echo "🎀 START TIME: $(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}"

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export TMPDIR=/scratch
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"


echo go $COUNT_NODE
echo $HOSTNAMES

start=$1
end=$2

launch_args="/fsx/loubna/projects/cosmopedia/prompts/judge/run_classification.py\
            --dataset_name /fsx/loubna/data/fineweb_2024_10 \
            --start $start \
            --end $end \
            --classifier_path /fsx/loubna/projects/cosmopedia/prompts/judge/embeddings_cache/logistic_regression_binary_classifier.joblib"

srun  -u bash -c "python3 ${launch_args}"

start=$(date +%s)
echo "🎀 END TIME: $(date -d @${start} "+%Y-%m-%d %H:%M:%S")"