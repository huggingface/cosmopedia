import os

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter


# you can also change ngrams or the number of buckets and their size here
minhash_config = MinhashConfig()  
HF_DATA = "cosmopedia-100k"

S3_MINHASH_BASE_PATH = f"s3://synthetic-datasets-phi/{HF_DATA}/minhash"
S3_LOGS_FOLDER = f"s3://synthetic-datasets-phi/{HF_DATA}/minhash_logs/"
LOCAL_LOGS_FOLDER = f"./logs/dedup_extras/{HF_DATA}"
os.makedirs(LOCAL_LOGS_FOLDER, exist_ok = True) 

TOTAL_TASKS = 120

INPUT_READER  = HuggingFaceDatasetReader(
    dataset=f"HuggingFaceTB/{HF_DATA}",  # dataset name
    dataset_options={
        "split": "train"
    },
    text_key="completion"
)
# stage 1 computes minhash signatures for each task (each task gets a set of files)
stage1 = SlurmPipelineExecutor(
    job_name="mh1",
    pipeline=[
        INPUT_READER,
        MinhashDedupSignature(output_folder=f"{S3_MINHASH_BASE_PATH}/signatures", config=minhash_config),
    ],
    tasks=TOTAL_TASKS,
    time="5:00:00",
    partition="hopper-cpu",
    logging_dir=f"{S3_LOGS_FOLDER}/signatures",
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/signatures/slurm_logs",
    qos="high",
)

# stage 2 finds matches between signatures in each bucket
stage2 = SlurmPipelineExecutor(
    job_name="mh2",
    pipeline=[
        MinhashDedupBuckets(
            input_folder=f"{S3_MINHASH_BASE_PATH}/signatures",
            output_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
            config=minhash_config,
        ),
    ],
    tasks=minhash_config.num_buckets,
    time="90:00:00",
    partition="hopper-prod",
    logging_dir=f"{S3_LOGS_FOLDER}/buckets",
    depends=stage1,
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/buckets/slurm_logs",
    qos="high",
)

# stage 3 creates clusters of duplicates using the results from all buckets
stage3 = SlurmPipelineExecutor(
    job_name="mh3",
    pipeline=[
        MinhashDedupCluster(
            input_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
            output_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
            config=minhash_config,
        ),
    ],
    tasks=1,
    time="90:00:00",
    partition="hopper-prod",
    logging_dir=f"{S3_LOGS_FOLDER}/clusters",
    mem_per_cpu_gb=70,
    cpus_per_task=2,
    depends=stage2,
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/clusters/slurm_logs",
)

# stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
# the data must match exactly stage 1, so number of tasks and the input source must be the same
stage4 = SlurmPipelineExecutor(
    job_name="mh4",
    pipeline=[
        INPUT_READER,
        TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
        MinhashDedupFilter(
            input_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
            exclusion_writer=JsonlWriter(f"{S3_MINHASH_BASE_PATH}/removed"),
        ),
        JsonlWriter(output_folder=f"{S3_MINHASH_BASE_PATH}/deduplicated_output"), # output_folder="hf_stack"
    ],
    tasks=TOTAL_TASKS,
    time="50:00:00",
    partition="hopper-cpu",
    logging_dir=f"{S3_LOGS_FOLDER}/filter",
    depends=stage3,
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/filter/slurm_logs",
)


stage4.run()
