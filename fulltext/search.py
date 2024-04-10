import json
import time
import re
import sys

import requests
from datasets import load_dataset


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


def add_hits(sample):
    topics = re.findall(r'\d+\.\s*(.+?)(?=\n\s*\d+\.|$)', sample["completion"], re.DOTALL)
    topics = [topic.strip() for topic in topics]
    sample["topics"] = topics

    sample["topic_hits"] = []
    for topic in topics:
        query = ' '.join(topic.split()[:32])
        while True:
            try:
                response = requests.post(
                    "http://127.0.0.1:9308/search",
                    data=json.dumps({
                        "index": "fineweb",
                        "size": 20,
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


data = load_dataset("HuggingFaceTB/bisac_topics_expanded_2", split="train", cache_dir="/scratch/cosmo/.cache")
data = data.select(range(16)).map(add_hits)
data.push_to_hub("HuggingFaceTB/bisac_topics_expanded_2_with_search_sample", private=True)
