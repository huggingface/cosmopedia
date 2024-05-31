from datasets import load_dataset


data = load_dataset("HuggingFaceTB/fineweb_edu_textless", "default", split="train", cache_dir="/fsx/anton/cosmo/cache/", num_proc=64)
data = data.select_columns(["token_count", "int_score"])
d2 = data.filter(lambda x: x["int_score"] >= 2, num_proc=64)
print("2: ", sum(d2["token_count"]))
d3 = d2.filter(lambda x: x["int_score"] >= 3, num_proc=64)
print("3: ", sum(d3["token_count"]))
d4 = d3.filter(lambda x: x["int_score"] >= 4, num_proc=64)
print("4: ", sum(d4["token_count"]))
d5 = d4.filter(lambda x: x["int_score"] >= 5, num_proc=64)
print("5: ", sum(d5["token_count"]))