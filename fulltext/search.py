import argparse
import json
import os
import sys
import time

import pandas as pd
import requests
from datasets import Dataset, load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, default="HuggingFaceTB/bisac_expanded_final")
    parser.add_argument("--n_topics", type=int, default=-1)
    parser.add_argument("--n_pages", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--target_datadet_prefix", type=str, default="bisac_boosted_new_index")
    parser.add_argument("--shuffle_seed", type=int, default=0)
    return parser.parse_args()


# wait until the server is up
while True:
    try:
        requests.post(
            "http://127.0.0.1:9308/search",
            data='{"index": "fineweb", "query": {"match": {"content": "ping"}}}',
        )
        break
    except requests.exceptions.ConnectionError:
        time.sleep(10)
        pass


args = get_args()
data = load_dataset(args.input_dataset, split="train", cache_dir="/scratch/cosmo/.cache")

data = data if args.n_topics < 0 else data.shuffle(args.shuffle_seed).select(range(args.n_topics))
print(f"Dataset with topics: {data}")

# path where the data chunks will be saved
data_path = f"{args.target_datadet_prefix}/{args.n_topics}t_{args.n_pages}p/data"
os.makedirs(data_path, exist_ok=True)

intermediate_data = []
total_length = len(data)
for index in range(len(data)):
    sample = data[index]
    sample["topic_hits"] = []
    top_category = sample["top_category"].strip()
    subcategory = sample["subcategory"].strip()
    subtopic = sample["subtopic"].strip()
    for c in ['!', '"', '$', "'", '(', ')', '/', '<', '@', '\\', '^', '|', '~']:
        top_category = top_category.replace(c, ' ')
        subcategory = subcategory.replace(c, ' ')
        subtopic = subtopic.replace(c, ' ')
    # boosting the IDF score of subtopic tokens
    subtopic = " ".join([w + "^2" for w in subtopic.split()])
    query = " ".join([top_category, subcategory, subtopic])
    while True:
        try:
            max_pages = 3_000
            print(f"n_pages requested: {args.n_pages}, max_pages: {max_pages}")
            response = requests.post(
                "http://127.0.0.1:9308/search",
                data=json.dumps(
                    {
                        "index": "fineweb",
                        "size": args.n_pages,
                        "query": {"query_string": query},
                        "max_matches": max_pages,
                    }
                ),
                timeout=1000,
            )
            if response.status_code != 200:
                print(response.text, file=sys.stderr)
                time.sleep(5)
                continue
            else:
                hits = response.json()["hits"]["hits"]
                sample["topic_hits"] = hits
                print(f"Number pages retrieved: {len(hits)} for query {query}")
                break
        except requests.exceptions.ConnectionError as e:
            print(e, file=sys.stderr)
            time.sleep(5)
            continue
    intermediate_data.append(sample)

    # Save data every save_interval topics and reinitialize intermediate dicts
    save_interval = min(args.n_topics, args.save_interval)
    if (index > 0 and (index + 1) % save_interval == 0) or (index + 1 == total_length):
        start_index = index + 1 - save_interval if (index + 1) % save_interval == 0 else max(0, index + 1 - (index + 1) % save_interval)
        save_path = f"{data_path}/data_{start_index}_{index + 1}.parquet"
        # we load in a dataframe first to explode topic hits and avoid OOM with `datasets`
        print(f"Saving data {index}")
        df = pd.DataFrame(intermediate_data).explode("topic_hits")
        ds = Dataset.from_pandas(df)
        ds.to_parquet(save_path)
        print(f"💾 Saved intermediate data at {save_path}")
        intermediate_data = []

print(f"Done! The data is available at '{save_path}' 🔥")
