import argparse
import logging
import os
import pickle

import numpy as np
from datasets import load_dataset
from joblib import load
from sentence_transformers import LoggingHandler

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier_path",
        type=str,
        default="/fsx/loubna/projects/cosmopedia/prompts/judge/embeddings_cache/logistic_regression_binary_classifier.joblib",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceTB/cosmopedia_web_textbooks",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/fsx/loubna/projects/cosmopedia/prompts/judge/embeddings_cache",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1_000_000)
    parser.add_argument("--target_org", type=str, default="HuggingFaceTB")
    parser.add_argument("--override_embedding_path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    format_number = (
        lambda num: f"{num//1_000_000}M" if num >= 1_000_000 else f"{num//1_000}k"
    )
    if args.override_embedding_path:
        logging.info(f"An embeddings path was provided: {args.override_embedding_path}")
        cache_file = args.override_embedding_path
    else:
        # embeddings_file_name = f"{args.dataset_name.split('/')[-1]}_{format_number(args.start)}_{format_number(args.end)}"
        embeddings_file_name = f"{args.dataset_name.split('/')[1]}_{format_number(args.start)}_{format_number(args.end)}"
        cache_file = os.path.join(args.cache_dir, f"embed_{embeddings_file_name}.pkl")
    assert os.path.exists(cache_file), f"No embeddings found at {cache_file}."

    logging.info(f"📚 Loading embeddings for dataset {args.dataset_name} from {args.start} to {args.end}")
    with open(cache_file, "rb") as f:
        embeddings = pickle.load(f)
    dataset = load_dataset(args.dataset_name, split="train", num_proc=48)
    dataset = dataset.select(range(args.start, args.end))
    assert len(embeddings) == len(dataset), f"wrong size in the embeddings"

    logging.info("🤖 Running inference with the classifier...")
    pipeline = load(args.classifier_path)
    X = np.array(embeddings)
    Y_pred = pipeline.predict(X)
    print(f"Percentage of positives: {(sum(Y_pred) * 100)/len(Y_pred):.2f}%")

    dataset = dataset.add_column("predicted_class", Y_pred)
    logging.info(dataset)
    dataset.push_to_hub(f"{args.target_org}/{embeddings_file_name}", private=True)
    logging.info(
        f"🎀 Inference done! Data saved at: {args.target_org}/{embeddings_file_name}"
    )
