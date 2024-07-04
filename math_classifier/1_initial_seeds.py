from datasets import load_dataset

owm_data = load_dataset("open-web-math/open-web-math", split="train", cache_dir="/fsx/anton/.cache/", num_proc=60)
fw_data = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", cache_dir="/fsx/anton/.cache/", num_proc=60)

owm_data = owm_data.select_columns(["url", "text"])
fw_data = fw_data.select_columns(["url", "text"])

# 500_000 train + 50_000 test
owm_data = owm_data.shuffle(seed=0).select(range(550_000))
fw_data = fw_data.shuffle(seed=0).select(range(550_000))

owm_data.push_to_hub("HuggingFaceTB/math_classifier_seeds", "owm", max_shard_size="2048MB", private=True)
fw_data.push_to_hub("HuggingFaceTB/math_classifier_seeds", "fineweb-raw", max_shard_size="2048MB", private=True)