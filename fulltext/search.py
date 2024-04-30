import json
import time
import re
import sys
import argparse
import random

import requests
from datasets import load_dataset


N_PAGES = 1000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, default="HuggingFaceTB/bisac_expanded_final")
    parser.add_argument("--n_topics", type=int, default=2000)
    parser.add_argument("--n_pages", type=int, default=1000)
    return parser.parse_args()

def sample_pages(x):
    return {"sample_100_pages": x["topic_hits"][:100]}


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


def get_query(sample):
    topics = re.findall(r'\d+\.\s*(.+?)(?=\n\s*\d+\.|$)', sample["completion"], re.DOTALL)
    topics = [topic.strip() for topic in topics]
    topics_only = [topic.split(":")[0].strip("* ") for topic in topics]
    return {"topics_only": topics_only, "topics": topics}


def add_hits(sample, n_pages):
    sample["topic_hits"] = []
    query = " / ".join([sample["top_category"].strip(), sample["subcategory"].strip(), sample["subtopic"].strip()])
    sample["query"] = query
    print(f"Query: {query}")
    while True:
        try:
            response = requests.post(
                "http://127.0.0.1:9308/search",
                data=json.dumps({
                    "index": "fineweb",
                    "size": n_pages,
                    "query": {
                        "match": {
                            "content": query
                        }
                    }
                }),
                timeout=1000
            )
            if response.status_code != 200:
                print(response.text, file=sys.stderr)
                time.sleep(5)
                continue
            else:
                hits = response.json()["hits"]["hits"]
                sample["topic_hits"].append(hits)
                break
        except requests.exceptions.ConnectionError as e:
            print(e, file=sys.stderr)
            time.sleep(5)
            continue

    return sample

args = get_args()
data = load_dataset(args.input_dataset, split="train", cache_dir="/scratch/cosmo/.cache")
data = data.shuffle(0).select(range(args.n_topics))
data = data.map(add_hits, fn_kwargs={"n_pages": args.n_pages})
subset = data.map(lambda x: {"sample_100_pages": x["topic_hits"][:100]})

name = f"search_bisac_{args.n_topics}_topics"
name_subset = name + "_subset_20"
subset.push_to_hub(f"HuggingFaceTB/{name_subset}", private=True)
print("Subset saved 🧸")
data.push_to_hub(f"HuggingFaceTB/{name}", private=True)
print("Full data saved 🧸")

