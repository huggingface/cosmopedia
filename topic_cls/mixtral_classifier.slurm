#!/bin/bash
#SBATCH --job-name=mixtral_classifier
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=21G
#SBATCH --gpus=4
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=7-00:00:00

set -x -e
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
source activate tgi

export volume="/scratch/cosmo/.cache"
mkdir -p "$volume"
export model="mistralai/Mistral-7B-Instruct-v0.2"

function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
export PORT=$(unused_port)
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    # try reading from file
    export HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)
fi
echo "Starting TGI container port $PORT"

# unset cache dirs to avoid pyxis having host env var somehow get into the container
unset HF_HUB_CACHE HF_ASSETS_CACHE HF_DATASETS_CACHE HF_MODULES_CACHE
srun --container-image='ghcr.io#huggingface/text-generation-inference' \
    --container-env=HUGGING_FACE_HUB_TOKEN,PORT \
    --container-mounts="$volume:/data" \
    --no-container-mount-home \
    --qos normal \
    /usr/local/bin/text-generation-launcher \
    --model-id $model \
    --max-concurrent-requests 2000 \
    --max-total-tokens 32768 \
    --max-input-length 32668 \
    --max-batch-prefill-tokens 32768 &

python mixtral_classifier.py --port $PORT

sleep 1000000000
