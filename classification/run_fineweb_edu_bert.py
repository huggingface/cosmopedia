import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import s3fs
import gzip
import json
import sys
from tqdm import tqdm


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def compute_scores(batch):
        texts = [record["text"] for record in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).float().cpu().numpy()

        for record, score in zip(batch, logits.tolist()):
            record["score"] = score
            record["int_score"] = int(round(max(0, min(score, 5))))
        return batch

    s3 = s3fs.S3FileSystem()
    file_paths = sorted(s3.glob(args.input_path), reverse=True)

    # select every arg.shard-th file
    file_paths = [file_paths[i] for i in range(args.shard, len(file_paths), args.num_shards)]
    for input_file_path in file_paths:
        output_file_path = input_file_path.replace(args.input_path.split("*")[0], args.output_path)
        tmp_file_path = input_file_path.replace(args.input_path.split("*")[0], args.tmp_path)
        if s3.exists(output_file_path):
            print(f"File {output_file_path} already exists", file=sys.stderr)
            continue

        with gzip.open(s3.open(input_file_path, "rb"), "rt") as fin, gzip.open(s3.open(tmp_file_path, "wb"), "wt") as fout:
            print(f"Processing {input_file_path}", file=sys.stderr)
            rows = []
            for line in tqdm(fin, desc=input_file_path, unit=" pages"):
                batch.append(json.loads(line))
                if len(batch) == 512:
                    batch = compute_scores(batch)
                    for record in batch:
                        fout.write(json.dumps(record) + "\n")
                    batch = []
            if batch:
                batch = compute_scores(batch)
                for record in batch:
                    fout.write(json.dumps(record) + "\n")

        s3.mv(tmp_file_path, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/snowflake_m_edu_reg")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tmp_path", type=str, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)

    args = parser.parse_args()
    main(args)
