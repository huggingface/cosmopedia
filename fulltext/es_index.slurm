#!/bin/bash
#SBATCH --job-name=index_fineweb
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=21G
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=7-00:00:00

set -x -e
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
source activate pyspark

ulimit -n 65535

mkdir -p /scratch/cosmo/es_data
rm -rf /scratch/cosmo/es_data/*
chmod -R 777 /scratch/cosmo/es_data

srun --container-image='elasticsearch:8.13.0' \
    --container-env="ES_JAVA_OPTS=-Djava.rmi.server.hostname=localhost -Dcom.sun.management.jmxremote=false" \
    --container-mounts="/sys/fs/cgroup:/sys/fs/cgroup:ro,/scratch/cosmo/es_data:/usr/share/elasticsearch/data" \
    --container-writable \
    --no-container-mount-home \
    --container-remap-root \
    --qos normal \
    /usr/local/bin/docker-entrypoint.sh eswrapper &

sleep 10000000000