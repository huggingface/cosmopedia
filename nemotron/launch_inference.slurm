#!/bin/bash
#SBATCH --job-name=nemotron
#SBATCH --partition hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=01:00:00

set -x
source ~/.bashrc

RESULTS="/fsx/anton/repos/nemotron"
OUTFILE="${RESULTS}/logs/slurm-%j-%n.out"
ERRFILE="${RESULTS}/logs/error-%j-%n.out"
MODEL="/fsx/anton/nemotron/Nemotron-4-340B-Instruct"
CONTAINER="nvcr.io/nvidia/nemo:24.01.framework"
MOUNTS="--container-mounts=${RESULTS}:/scripts,${MODEL}:/model"

read -r -d '' cmd <<EOF
bash /scripts/nemo_inference.sh /model
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"