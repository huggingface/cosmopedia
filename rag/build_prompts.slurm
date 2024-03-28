#!/bin/bash
#SBATCH --job-name=rag_prompts
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=21G
#SBATCH --gpus=1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -x -e
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
source activate lighteval

mkdir -p "/scratch/cosmo/qdrant_storage"

srun --container-image='qdrant/qdrant:v1.8.2' \
    --container-mounts="/scratch/cosmo/qdrant_storage:/qdrant/storage:z,/fsx/anton/qdrant_snapshots:/qdrant/snapshots:z" \
    --no-container-mount-home \
    --qos high \
    /qdrant/entrypoint.sh &

python build_prompts.py
