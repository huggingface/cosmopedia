import argparse
import logging
import os
import pickle

from datasets import load_dataset
from sentence_transformers import LoggingHandler, SentenceTransformer


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="HuggingFaceTB/cosmopedia_web_textbooks"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/fsx/loubna/projects/cosmopedia/prompts/judge/embeddings_cache",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1_000_000)
    parser.add_argument("--text_col", type=str, default="text")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.info(f"📚 Loading dataset {args.dataset_name} from {args.start} to {args.end}")
    dataset = load_dataset(args.dataset_name, split="train", num_proc=48)
    dataset = dataset.select(range(args.start, args.end))
    sentences = dataset[args.text_col]

    format_number = (
        lambda num: f"{num//1_000_000}M" if num >= 1_000_000 else f"{num//1_000}k"
    )
    embeddings_file_name = f"{args.dataset_name.split('/')[-1]}_{format_number(args.start)}_{format_number(args.end)}"
    cache_file = os.path.join(args.cache_dir, f"embed_{embeddings_file_name}.pkl")
    logging.info(f"Save path {cache_file}")

    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=512)
    # uses all available CUDA devices
    pool = model.start_multi_process_pool()
    embeddings = model.encode_multi_process(sentences, pool)
    logging.info("⚙️ Embeddings computed. Shape: {}".format(embeddings.shape))

    model.stop_multi_process_pool(pool)

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    with open(cache_file, "wb") as f:
        pickle.dump(embeddings, f)
    logging.info(f"💾 Embeddings saved at {cache_file}")
