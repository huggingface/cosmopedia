#!/bin/bash
#SBATCH --job-name=filter_edu_fineweb_2
#SBATCH --partition hopper-cpu
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -o /fsx/anton/logs/%x_%j.out
#SBATCH -e /fsx/anton/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --array=0-94%16

set -x -e
source ~/.bashrc
source "/admin/home/anton/miniforge3/etc/profile.d/conda.sh"
source activate lighteval

readarray -t all_dumps < <(aws s3 ls s3://cosmopedia-data/fineweb_edu_scores/ | awk '{print $2}')
dump_path=${all_dumps[${SLURM_ARRAY_TASK_ID}]}
# e.g. "CC-MAIN-2013-20/"

python filter_edu_fineweb.py \
    --input_path="cosmopedia-data/fineweb_edu_scores/${dump_path}" \
    --output_dataset="HuggingFaceTB/fineweb_edu_2" \
    --threshold=2
