from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import ParquetWriter, JsonlWriter, HuggingFaceDatasetWriter

# execut = SlurmPipelineExecutor(
#     pipeline=[
#         ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/", glob_pattern="CC-*/*.parquet"),
#         SamplerFilter(rate=0.266),
#         JsonlWriter("s3://cosmopedia-data/fineweb_edu_samples/350BT/"),
#         SamplerFilter(rate=0.286),
#         JsonlWriter("s3://cosmopedia-data/fineweb_edu_samples/100BT/"),
#         SamplerFilter(rate=0.1),
#         JsonlWriter("s3://cosmopedia-data/fineweb_edu_samples/10BT/")
#     ],
#     logging_dir="/fsx/anton/logs/fineweb-edu/sample-fineweb-edu/",
#     tasks=500,
#     qos="high",
#     randomize_start=True,
#     max_array_launch_parallel=True,
#     skip_completed=True,
#     mem_per_cpu_gb=4,
#     time="08:00:00",
#     partition="hopper-cpu"
# )
#
# execut.run()


for subset, tasks in zip((10, 100, 350), (1, 10, 30)):
    SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(f"s3://cosmopedia-data/fineweb_edu_samples/{subset}BT"),
            HuggingFaceDatasetWriter(
                "HuggingFaceFW/fineweb-edu",
                local_working_dir=f"/scratch/anton/upload_fw_edu/{subset}BT",
                private=False, max_file_size=2 * 2**30,
                output_filename=f"sample/{subset}BT/${{rank}}.parquet"
            )
        ],
        job_name=f"fineweb_up_{subset}BT",
        time="2:00:00",
        tasks=tasks,
        qos="high",
        logging_dir=f"/fsx/anton/logs/upload_fineweb_edu_samples/{subset}BT",
        partition="hopper-cpu",
        cpus_per_task=2,
        mem_per_cpu_gb=4,
        skip_completed=True,
        randomize_start=True,
    ).run()