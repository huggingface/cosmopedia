import re
from datasets import load_dataset

all_urls = re.compile(r'^(https?://)?(stackoverflow\.com|mathoverflow\.net|((math|physics|stats|english|diy|codereview|travel|money|law|photo|music|movies|politics|tex|dba|mathematica|datascience|dsp|cooking|gardening|codegolf|fitness|scicomp|psychology|medicalsciences|earthscience|linguistics|gaming|rpg|puzzling|ell|cstheory)\.stackexchange\.com))/.*')
math_code_urls = pattern = re.compile(r'^(https?://)(stackoverflow\.com|mathoverflow\.net|((math|physics|stats|codereview|tex|dba|mathematica|datascience|dsp|codegolf|scicomp|cstheory)\.stackexchange\.com))/questions/[0-9]+/.*')

dataset = load_dataset("HuggingFaceFW/fineweb", split="train",
                       data_files=["data/CC-MAIN-2023-23/*.parquet", "data/CC-MAIN-2023-40/*.parquet", "data/CC-MAIN-2023-50/*.parquet", "data/CC-MAIN-2024-10/*.parquet"],
                       cache_dir="/fsx/anton/.cache/", num_proc=60)
dataset = dataset.filter(lambda x: bool(all_urls.match(x["url"])), num_proc=60)

def label_fn(url):
    return {"label": 1} if math_code_urls.match(url) else {"label": 0}

dataset = dataset.map(label_fn, input_columns=["url"], num_proc=1)
dataset.push_to_hub("HuggingFaceTB/fineweb_math_seeds", max_shard_size="2048MB", private=True)
