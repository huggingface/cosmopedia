from datasets import load_dataset
import re
import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from functools import lru_cache


def clean_title(title):
    if title is None:
        return ""
    # remove chapter numbering
    title = re.sub(r"^(\d+\.)*\s*", "", title.strip())
    if not title.endswith((".", "?", "!")):
        title += "."
    return title


def transform_query(query):
    return f"Represent this sentence for searching relevant passages: {query}"


def pooling(outputs, inputs, strategy="cls"):
    if strategy == "cls":
        outputs = outputs[:, 0]
    elif strategy == "mean":
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1
        ) / torch.sum(inputs["attention_mask"])
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()


client = QdrantClient("localhost", grpc_port=6334, prefer_grpc=True, timeout=600)
client.recover_snapshot(
    "wikipedia",
    "file:///qdrant/snapshots/wikipedia/wikipedia-737832153632974-2024-03-19-18-23-18.snapshot",
)


model_id = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/scratch/cosmo/cache/")
data = load_dataset(
    "HuggingFaceTB/stanford_prompts_1M",
    split="train",
    cache_dir="/scratch/cosmo/cache/",
)
model = AutoModel.from_pretrained(model_id, cache_dir="/scratch/cosmo/cache/").to(
    "cuda"
)


@lru_cache(maxsize=100)
def get_passages(query):
    inputs = tokenizer([query], padding=True, return_tensors="pt").to("cuda")
    outputs = model(**inputs).last_hidden_state
    query_vector = pooling(outputs, inputs, "cls")[0]

    hits = client.search(
        collection_name="wikipedia", query_vector=query_vector.tolist(), limit=50
    )

    # remove hits where >80% of the passage's lines start with *:
    def list_fraction(text):
        lines = text.split("\n")
        return len([line for line in lines if line.strip().startswith("*")]) / len(lines)

    hits = [hit for hit in hits if hit.payload and list_fraction(hit.payload["text"]) < 0.8]

    return hits[:10]


def build_prompt(sample):
    course_title = clean_title(sample["course_title"])
    section = clean_title(sample["section"])
    unit = clean_title(sample["unit"])

    query = transform_query(f"{course_title} {section} {unit}")
    hits = get_passages(query)

    prompt = """RELEVANT PASSAGES:"""

    for hit in hits:
        prompt += (
            f"""\nArticle titled "{hit.payload['title']}":\n{hit.payload['text']}\n\n"""
        )
    prompt += (
        f"TASK:\nUsing the information from RELEVANT PASSAGES, w" + sample["prompt"][1:]
    )

    sample["rag_prompt"] = prompt
    sample["rag_query"] = f"{course_title} {section} {unit}"
    sample["rag_passages"] = [hit.payload for hit in hits]
    sample["rag_scores"] = [hit.score for hit in hits]

    return sample


data = data.sort(["course_title", "section", "unit"])
data = data.map(build_prompt)

data.push_to_hub("HuggingFaceTB/stanford_prompts_1M_rag", private=True)
