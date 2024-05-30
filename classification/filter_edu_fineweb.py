import argparse
import s3fs
import json
from datasets import load_dataset
import multiprocessing as mp
import time
from tqdm import tqdm
import os
from smart_open import open
import shutil
from huggingface_hub import HfFileSystem


def process_file(args):
    input_file_path, threshold = args
    output_file = "/fsx/anton/cosmo/" + input_file_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_tokens = 0
    kept_tokens = 0
    total_docs = 0
    kept_docs = 0
    with open("s3://" + input_file_path, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            record = json.loads(line)
            if record["int_score"] >= threshold:
                fout.write(json.dumps({
                    "text": record["text"],
                    "id": record["id"],
                    "dump": record["metadata"]["dump"],
                    "url": record["metadata"]["url"],
                    "file_path": record["metadata"]["file_path"],
                    "language": record["metadata"]["language"],
                    "language_score": record["metadata"]["language_score"],
                    "token_count": record["metadata"]["token_count"],
                    "score": record["score"],
                    "int_score": record["int_score"],
                }, ensure_ascii=False) + "\n")
                kept_docs += 1
                kept_tokens += record["metadata"]["token_count"]
            total_docs += 1
            total_tokens += record["metadata"]["token_count"]

    return total_docs, kept_docs, total_tokens, kept_tokens


def main(args):
    s3 = s3fs.S3FileSystem()
    input_files = s3.glob(f"{args.input_path}*.jsonl.gz")
    config_name = args.input_path.strip("/").split("/")[-1]

    hf = HfFileSystem()
    if hf.exists(f"datasets/{args.output_dataset}/{config_name}"):
        print(f"Config {config_name} already exists")
        return

    pool = mp.Pool(mp.cpu_count())
    results = pool.imap_unordered(process_file, [(file, args.threshold) for file in input_files])

    total_tokens = 0
    kept_tokens = 0
    total_docs = 0
    kept_docs = 0
    for total_docs_, kept_docs_, total_tokens_, kept_tokens_ in tqdm(results, total=len(input_files)):
        total_docs += total_docs_
        kept_docs += kept_docs_
        total_tokens += total_tokens_
        kept_tokens += kept_tokens_
    print(f"Total docs: {total_docs}, kept docs: {kept_docs}, percentage: {100*kept_docs/total_docs:.2f}%", flush=True)
    print(f"Total tokens: {total_tokens}, kept tokens: {kept_tokens}, percentage: {100*kept_tokens/total_tokens:.2f}%", flush=True)

    data = load_dataset("json", data_files=f"/fsx/anton/cosmo/{args.input_path}*.jsonl.gz",
                        cache_dir=f"/fsx/anton/cosmo/{args.input_path}cache/", num_proc=mp.cpu_count())
    while True:
        try:
            data.push_to_hub(args.output_dataset, config_name=config_name, private=True, max_shard_size="4096MB")
            break
        except Exception as e:
            print(e)
            time.sleep(1)
            continue

    shutil.rmtree(f"/fsx/anton/cosmo/{args.input_path}", ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dataset", type=str, required=True)
    parser.add_argument("--threshold", type=int, required=True)

    args = parser.parse_args()
    main(args)
