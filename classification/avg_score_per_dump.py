import s3fs
import json
import multiprocessing as mp
from tqdm import tqdm
from smart_open import open
from collections import defaultdict


def process_file(input_file_path):
    dump = input_file_path.split("/")[-2]

    total_score = 0
    n_pages = 0
    with open("s3://" + input_file_path, "r") as fin:
        for i, line in enumerate(fin):
            record = json.loads(line)
            total_score += record["score"]
            n_pages += 1
            if i > 10000:
                break

    return dump, total_score, n_pages


def main():
    s3 = s3fs.S3FileSystem()
    input_files = s3.glob(f"cosmopedia-data/fineweb_edu_scores/*/*.jsonl.gz")

    pool = mp.Pool(mp.cpu_count())
    results = pool.imap_unordered(process_file, input_files)

    dump_total_score = defaultdict(int)
    dump_n_pages = defaultdict(int)
    for dump, score, n_pages in tqdm(results, total=len(input_files)):
        dump_total_score[dump] += score
        dump_n_pages[dump] += n_pages

    dumps = sorted(dump_total_score.keys())
    for dump in dumps:
        avg_score = dump_total_score[dump] / dump_n_pages[dump]
        print(f"{dump}: {avg_score:.5f}")


if __name__ == "__main__":
    main()
