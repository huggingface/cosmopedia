import argparse
import difflib
import re
import unicodedata
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset


def tokenize(text):
    """Normalize text by removing diacritics and tokenize."""
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    tokens = re.findall("\w+", text.lower())
    return tokens


def get_ngrams(tokens, n):
    """Generate n-grams from tokens."""
    return set(zip(*[tokens[i:] for i in range(n)]))


def retrieve_ngrams_batch(batch, eval_ngrams, eval_datasets, eval_texts, ngram_len):
    """Find contaminated samples based on n-grams."""
    new_batch = {"completion": [], "ngram": [], "bench_name": [], "bench_text": []}
    for completion in batch["completion"]:
        tokens = tokenize(completion)
        ngrams = get_ngrams(tokens, ngram_len)
        for ngram in ngrams:
            if ngram in eval_ngrams:
                idx = eval_ngrams[ngram]
                new_batch["completion"].append(completion)
                new_batch["ngram"].append(ngram)
                new_batch["bench_name"].append(eval_datasets[idx])
                new_batch["bench_text"].append(eval_texts[idx])
                break
    return new_batch


def diff_strings(string1, string2):
    """Find matching parts between two strings."""
    matcher = difflib.SequenceMatcher(None, string1.lower(), string2.lower(), autojunk=False)
    matching_blocks = matcher.get_matching_blocks()
    matches = []
    for block in matching_blocks:
        start_a, start_b, length = block
        if length > 5:
            match = string1[start_a:start_a + length]
            matches.append(match)
    return matches


def add_match_stats(example):
    gen_text = " ".join(tokenize(example["completion"]))
    bench_text = " ".join(tokenize(example["bench_text"]))
    matching_parts = diff_strings(gen_text, bench_text)
    match = " ".join("".join(matching_parts).split())
    example["diff"] = matching_parts
    example["diff_ratio"] = len(match) / len(bench_text) if len(bench_text) > 0 else 0
    example["diff_length"] = len(match)
    example["longest_diff_part"] = max(matching_parts, key=len, default="")
    example["longest_diff_part_length"] = len(example["longest_diff_part"])
    return example


def main(args):
    # Load the evaluation data to build n-grams index
    eval_ngrams, eval_datasets, eval_texts = {}, [], []
    eval_data = load_dataset(args.eval_dataset, split="train")
    for example in tqdm(eval_data):
        tokens = tokenize(example["text"])
        ngrams = get_ngrams(tokens, args.ngram_length)
        if ngrams:
            idx = len(eval_texts)
            eval_ngrams.update(zip(ngrams, [idx] * len(ngrams)))
            eval_datasets.append(example.get("task_name", "unknown"))
            eval_texts.append(example["text"])

    train_dataset_path = Path(args.train_dataset)
    if train_dataset_path.exists() and train_dataset_path.suffix in [".json", ".csv"]:
        if train_dataset_path.suffix == ".json":
            train_data = Dataset.from_json(args.train_dataset)
        elif train_dataset_path.suffix == ".csv":
            train_data = Dataset.from_csv(args.train_dataset)
    else:
        train_data = load_dataset(args.train_dataset, split="train")

    contamination_report = train_data.map(
        lambda batch: retrieve_ngrams_batch(batch, eval_ngrams, eval_datasets, eval_texts, args.ngram_length),
        batched=True, batch_size=1000, num_proc=args.num_proc, remove_columns=train_data.column_names
    )

    contamination_report = contamination_report.map(
        lambda example: add_match_stats(example), num_proc=args.num_proc
    )

    contamination_report.push_to_hub(args.report_dataset_name, private=args.private)

    contamination_report = contamination_report.filter(lambda x: x["diff_ratio"] > args.diff_threshold)

    if args.save_decontaminated:
        contaminated_completions = set(contamination_report["completion"])
        filtered_data = train_data.filter(lambda x: x["completion"] not in contaminated_completions)
        filtered_data.push_to_hub(args.decontaminated_dataset_name, private=args.private)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a decontamination report for a dataset.")
    parser.add_argument("--eval_dataset", type=str,
                        default="HuggingFaceTB/phi2_eval_data_for_decontamination",
                        help="Name of the dataset with benchmark samples to use for decontamination.")
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path or name of the training dataset to process.")
    parser.add_argument("--report_dataset_name", type=str, required=True,
                        help="Name for the output dataset with decontamination report.")
    parser.add_argument("--decontaminated_dataset_name", type=str, help="Name for the decontaminated dataset.")
    parser.add_argument("--private", action='store_true', help="Whether to make the output dataset private.")
    parser.add_argument("--ngram_length", type=int, default=10, help="Length of the n-grams to consider.")
    parser.add_argument("--diff_threshold", type=float, default=0.5,
                        help="Threshold for filtering based on difference ratio.")
    parser.add_argument("--num_proc", type=int, default=90, help="Number of processes to use for map operations.")
    parser.add_argument("--save_decontaminated", action='store_true',
                        help="Whether to save the decontaminated dataset.")

    args = parser.parse_args()
    main(args)