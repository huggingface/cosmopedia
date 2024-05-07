import argparse
import json
import sys
import time

import requests
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, default="HuggingFaceTB/bisac_expanded_final")
    parser.add_argument("--n_pages", type=int, default=2000)
    parser.add_argument("--output_dataset", type=str, default="HuggingFaceTB/bisac_boosted_new_index_2000")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
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
data = data.filter(lambda x, i: i % args.num_shards == args.shard, with_indices=True)
data = data.select_columns(["top_category", "subcategory", "subtopic"])


def run_query(query, n_pages):
    while True:
        try:
            max_pages = 4_000
            response = requests.post(
                "http://127.0.0.1:9308/search",
                data=json.dumps(
                    {
                        "index": "fineweb",
                        "size": n_pages,
                        "query": query,
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
                return hits
        except requests.exceptions.ConnectionError as e:
            print(e, file=sys.stderr)
            time.sleep(5)
            continue


def search_topic(sample):
    top_category = sample["top_category"][0].strip()
    subcategory = sample["subcategory"][0].strip()
    subtopic = sample["subtopic"][0].strip()
    for c in ['!', '"', '$', "'", '(', ')', '/', '<', '@', '\\', '^', '|', '~']:
        top_category = top_category.replace(c, ' ')
        subcategory = subcategory.replace(c, ' ')
        subtopic = subtopic.replace(c, ' ')
    # boosting the IDF score of subtopic tokens
    boosted_subtopic = " ".join([w + "^2" for w in subtopic.split()])
    match_query = " ".join([top_category, subcategory, subtopic])
    boosted_query = " ".join([top_category, subcategory, boosted_subtopic])

    boosted_hits = run_query({"query_string": boosted_query}, args.n_pages)
    print(f"Boosted hits: {len(boosted_hits)} for {boosted_query}", file=sys.stderr)
    if len(boosted_hits) < args.n_pages:
        match_hits = run_query({"match": {"content": match_query}}, args.n_pages + len(boosted_hits))
        print(f"Match hits: {len(match_hits)} for {match_query}", file=sys.stderr)
    else:
        match_hits = []

    hit_ids = set()
    hits = []
    for hit in boosted_hits + match_hits:
        if hit["_id"] not in hit_ids:
            hits.append(hit)
            hit_ids.add(hit["_id"])
    hits = hits[:args.n_pages]

    results = {
        "top_category": sample["top_category"]*len(hits),
        "subcategory": sample["subcategory"]*len(hits),
        "subtopic": sample["subtopic"]*len(hits),
        "topic_hits": hits,
        "num_hits": [len(hits)]*len(hits),
    }
    return results


data = data.map(search_topic, batched=True, batch_size=1, num_proc=2)
data.push_to_hub(f"{args.output_dataset}_{args.shard}", private=True, max_shard_size="4096MB")
