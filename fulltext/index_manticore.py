import json
import time
import sys
import random

import requests
from datasets import load_dataset


def insert_batch(batch):
    """
    Function to insert a batch of records into Manticore Search
    """
    ndjson = ""

    index_name = f"fineweb{random.randint(0, 63)}"

    for text, _id, url, language_score, token_count in zip(
        batch["text"],
        batch["id"],
        batch["url"],
        batch["language_score"],
        batch["token_count"],
    ):
        doc = {
            "insert": {
                "index": index_name,
                "_id": _id.split(":")[-1].strip(">"),
                "doc": {
                    "content": text,
                    "fw_id": _id.split(":")[-1].strip(">"),
                    "url": url,
                    "language_score": language_score,
                    "token_count": token_count,
                },
            }
        }
        ndjson += json.dumps(doc) + "\n"

    response = None
    while response is None:
        try:
            response = requests.post(
                "http://127.0.0.1:9308/bulk",
                headers={"Content-Type": "application/x-ndjson"},
                data=ndjson,
            )
        except requests.exceptions.ConnectionError as e:
            print(e, file=sys.stderr)
            time.sleep(1)
            pass

    return {"response": [response.status_code]}


def main():
    sql_url = "http://127.0.0.1:9308/sql?mode=raw"

    print("Removing table", file=sys.stderr)
    while True:
        try:
            requests.post(sql_url, data={"query": "drop table if exists fineweb"})
            break
        except requests.exceptions.ConnectionError as e:
            print(e, file=sys.stderr)
            time.sleep(5)
            pass


    print("Creating table", file=sys.stderr)
    for i in range(64):
        response = requests.post(sql_url, data={"query": f"drop table if exists fineweb{i}"})
        print(response.text, file=sys.stderr)
        local_query = f"create table fineweb{i}(content text, fw_id string, url string, language_score float, token_count int) charset_table='non_cjk' stopwords='en' morphology='stem_en'"
        response = requests.post(sql_url, data={"query": local_query})
        print(response.text, file=sys.stderr)

    # Then, construct and execute the distributed table creation query
    distributed_query = "create table fineweb type='distributed'"
    for i in range(64):
        distributed_query += f" local='fineweb{i}'"
    response = requests.post(sql_url, data={"query": distributed_query})
    print(response.text, file=sys.stderr)

    for dump in ["CC-MAIN-2024-10", "CC-MAIN-2023-50"]:
        print("Loading dataset", file=sys.stderr)
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            dump,
            split="train",
            num_proc=64,
            cache_dir="/scratch/cosmo/.cache",
        )
        dataset = dataset.select_columns(
            ["text", "id", "url", "language_score", "token_count"]
        )
        dataset = dataset.map(
            insert_batch,
            batched=True,
            batch_size=10000,
            remove_columns=["text", "id", "url", "language_score", "token_count"],
            num_proc=64,
        )
        for _ in dataset:
            pass

    time.sleep(30)
    for i in range(64):
        print(f"Optimizing table fineweb{i}", file=sys.stderr)
        response = requests.post(
            sql_url,
            data={"query": f"FLUSH TABLE fineweb{i}"},
            timeout=600,
        )
        print(response.text, file=sys.stderr)
        response = requests.post(
            sql_url,
            data={"query": f"OPTIMIZE TABLE fineweb{i} OPTION cutoff=16, sync=1"},
            timeout=600,
        )
        print(response.text, file=sys.stderr)
        response = requests.post(
            sql_url,
            data={"query": f"FREEZE fineweb{i}"},
            timeout=600,
        )
        print(response.text, file=sys.stderr)

    response = requests.post(
        "http://127.0.0.1:9308/search",
        data='{"index":"fineweb","query":{"match":{"*":"hello world"}}}',
    )
    print(response.text, file=sys.stderr)

    # print("Backing up the index", file=sys.stderr)
    # time.sleep(30)
    # response = requests.post(
    #     sql_url,
    #     data={"query": "BACKUP TO /tmp/backups"},
    # )
    # print(response.text, file=sys.stderr)


if __name__ == "__main__":
    main()
