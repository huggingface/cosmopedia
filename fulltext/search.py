import argparse
import json
import sys
import time
import os

import requests
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, default="HuggingFaceTB/bisac_expanded_final")
    parser.add_argument("--n_topics", type=int, default=2000)
    parser.add_argument("--n_pages", type=int, default=1000)
    parser.add_argument("--n_chunks", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--target_datadet_prefix", type=str, default="HuggingFaceTB/search")
    return parser.parse_args()


def load_and_merge(directory_path, id, args):
    """To avoid datasets pyarrow overflow:
    1- load each json chunk in a pandas dataframe
    2- explode the topic_hits column
    3- convert each chunk to datasets and concatenate the subsets"""
    json_files = [file for file in os.listdir(directory_path) if file.endswith('.json')]
    print(f"Found {len(json_files)} chunks (={args.n_topics / args.save_interval - 1})")

    print("Loading the chunks...")
    data_subsets = []
    for file in tqdm(json_files):
        file_path = os.path.join(directory_path, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df = df.explode("topic_hits")
        data_subsets.append(Dataset.from_pandas(df))

    print("Merging the chunks...")
    merged_data = concatenate_datasets(data_subsets).remove_columns(["__index_level_0__"])
    print(merged_data)
    print(f"Sanity check on the pages from topic 0: {merged_data[0]['topic_hits']}")

    merged_data.push_to_hub(f"{args.target_datadet_prefix}_{id}", private=True)
    print(f"Done! The data is available at '{args.target_datadet_prefix}_{id}' 🔥")


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
data = data.shuffle(0).select(range(args.n_topics))
# path where the data chunks will be saved
id = f"{args.n_topics}t_{args.n_pages}p"
os.makedirs(f"intermediate_data/{id}", exist_ok=True)

intermediate_data = []
for index in range(len(data)):
    sample = data[index]
    sample["topic_hits"] = []
    query = " / ".join([sample["top_category"].strip(), sample["subcategory"].strip(), sample["subtopic"].strip(),])
    while True:
        try:
            response = requests.post(
                "http://127.0.0.1:9308/search",
                data=json.dumps(
                    {
                        "index": "fineweb",
                        "size": args.n_pages,
                        "query": {"match": {"content": query}},
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
                break
        except requests.exceptions.ConnectionError as e:
            print(e, file=sys.stderr)
            time.sleep(5)
            continue
    intermediate_data.append(sample)

    save_interval = min(args.n_topics, args.save_interval)
    if index > 0 and (index + 1) % save_interval == 0:
        # Save data every save_interval topics and reinitialize intermediate dicts
        save_path = f"intermediate_data/{id}/data_{index - save_interval}_{index}.json"
        with open(save_path, "w") as fp:
            json.dump(intermediate_data, fp)
        print(f"💾 Saved intermediate data at {save_path}")
        intermediate_data = []

# merge dataset and push to hub
directory_path = f"intermediate_data/{id}"
load_and_merge(directory_path, id, args)
