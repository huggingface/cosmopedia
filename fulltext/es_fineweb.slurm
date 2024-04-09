#!/bin/bash
#SBATCH --job-name=index_fineweb
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=21G
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=7-00:00:00

set -x -e
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
source activate pyspark

mkdir -p /scratch/cosmo/es_idx
mkdir -p /fsx/anton/es_backups
srun --container-image='elasticsearch:8.13.0' \
    --container-env="discovery.type=single-node,xpack.security.enabled=false" \
    --container-mounts="/sys/fs/cgroup:/sys/fs/cgroup:ro,/scratch/cosmo/es_idx:/usr/share/elasticsearch/data,/fsx/anton/es_backups:/usr/share/elasticsearch/backups:z" \
    --no-container-mount-home \
    --container-remap-root \
    --container-writable \
    --qos normal \
    /usr/local/bin/docker-entrypoint.sh eswrapper &

python index_es.py

sleep 1000000000
