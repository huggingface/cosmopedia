#!/bin/bash
#SBATCH --job-name=cosmo_search_sharded
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=20G
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --array=0-15%8

set -x -e
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
source activate pyspark

ulimit -n 99999

mkdir -p /scratch/cosmo/manticore_idx
rm -rf /scratch/cosmo/manticore_idx/*
rclone copy -P --transfers 32 s3:cosmopedia-data/manticore_idx/CC-MAIN-2024-10-2023-50/ /scratch/cosmo/manticore_idx/

srun --container-image='manticoresearch/manticore:6.2.12' \
    --container-mounts="/scratch/cosmo/manticore_idx:/var/lib/manticore:z,$(pwd)/manticore.conf:/etc/manticoresearch/manticore.conf" \
    --no-container-mount-home \
    --qos high \
    /bin/bash -c 'mkdir -p /var/run/manticore && chown manticore:manticore /var/run/manticore && mkdir -p /var/run/mysqld && chown manticore:manticore /var/run/mysqld && /entrypoint.sh searchd -c /etc/manticoresearch/manticore.conf --nodetach' &

python search_sharded.py --shard ${SLURM_ARRAY_TASK_ID} --num_shards 16