#!/bin/bash
#SBATCH --job-name=qdrant
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=11G
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -x -e
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
source activate lighteval

mkdir -p "/scratch/anton/qdrant_storage"

srun --container-image='qdrant/qdrant:v1.8.2' \
    --container-mounts="/scratch/anton/qdrant_storage:/qdrant/storage:z,/fsx/anton/qdrant_snapshots:/qdrant/snapshots:z" \
    --no-container-mount-home \
    --qos normal \
    /qdrant/entrypoint.sh &

python build_index.py

sleep 10m