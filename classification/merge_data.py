from datasets import load_dataset, concatenate_datasets
from shard import save_manual_shards

END = 12_000_000
STEP = 2_000_000
DATASET_NAME = "/fsx/loubna/data/fineweb_2024_10"

format_number = (
    lambda num: f"{num//1_000_000}M" if num >= 1_000_000 else f"{num//1_000}k"
)

dataset_list = []
for i in range(0, END, STEP):
    embeddings_file_name = f"{DATASET_NAME.split('/')[-1]}_{format_number(i)}_{format_number(i + STEP)}"
    save_path = f"/fsx/loubna/projects/cosmopedia/prompts/judge/data_temp/{embeddings_file_name}"
    print(f"Loading dataset ({i} - {i+STEP}) from {save_path}")
    ds = load_dataset(save_path, split="train", num_proc=48)
    dataset_list.append(ds)

final_data = concatenate_datasets(dataset_list)
print(final_data)
save_manual_shards(final_data, "/fsx/loubna/data/fineweb_2024_10_classification")