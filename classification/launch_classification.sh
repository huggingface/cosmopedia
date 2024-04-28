#!/bin/bash

# increment step for each job
INCREMENT=2000000

# maximum end value
MAX_END=12000000
#MAX_END=12000000

for ((start=0; start<MAX_END; start+=INCREMENT)); do
    end=$((start+INCREMENT))
    echo 🚀 Launching classification for samples from $start to $end
    sbatch /fsx/loubna/projects/cosmopedia/prompts/judge/run_classification.slurm "$start" "$end"
done