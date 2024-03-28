from itertools import islice
from qdrant_client import models, QdrantClient
from datasets import load_dataset
import hashlib
import uuid
import time


batch_size = 128
dataset = load_dataset(
    "anton-l/wiki-embed-mxbai-embed-large-v1",
    split="train",
    num_proc=32,
    cache_dir="/scratch/anton/.cache/",
)

client = QdrantClient("localhost", grpc_port=6334, prefer_grpc=True, timeout=600)

client.create_collection(
    collection_name="wikipedia",
    vectors_config=models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE,
        on_disk=True,
    ),
    optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
    hnsw_config=models.HnswConfigDiff(on_disk=True),
)


def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch


def index_batch(batch):
    client = QdrantClient("localhost", port=6333)
    ids = [
        str(
            uuid.uuid5(
                uuid.NAMESPACE_OID, hashlib.sha1(_id.encode("utf-8")).hexdigest()
            )
        )
        for _id in batch["id"]
    ]
    vectors = batch.pop("embeddings")
    payloads = [
        {k: batch[k][i] for k in batch.keys()} for i in range(len(batch["text"]))
    ]

    client.upsert(
        collection_name="wikipedia",
        points=models.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        ),
    )
    return {}


dataset.map(
    index_batch,
    batched=True,
    batch_size=batch_size,
    remove_columns=dataset.column_names,
    num_proc=8,
)

while client.get_collection(collection_name="wikipedia").status != "green":
    print("Waiting for the indexing to complete...")
    time.sleep(30)

hits = client.search(collection_name="wikipedia", query_vector=[0.0] * 1024, limit=5)

for hit in hits:
    print("score:", hit.score, hit.payload)

client.create_snapshot(collection_name="wikipedia")
