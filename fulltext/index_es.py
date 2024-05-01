from elasticsearch import Elasticsearch, helpers
from datasets import load_dataset

es = Elasticsearch("http://localhost:9200")


def create_index(index_name="fineweb"):
    """
    Function to create an index with specific mappings and settings for text analysis.
    """
    settings = {
        "settings": {
            "number_of_shards": 32,
            "number_of_replicas": 1,
            "analysis": {
                "filter": {
                    "english_stop": {
                        "type": "stop",
                        "stopwords": "_english_"
                    },
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    },
                    "english_possessive_stemmer": {
                        "type": "stemmer",
                        "language": "possessive_english"
                    }
                },
                "analyzer": {
                    "rebuilt_english": {
                        "tokenizer": "standard",
                        "filter": [
                            "english_possessive_stemmer",
                            "lowercase",
                            "english_stop",
                            "english_stemmer"
                        ]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "rebuilt_english",
                    "fielddata": True
                },
                "url": {"type": "keyword"},
                "language_score": {"type": "float"},
                "token_count": {"type": "integer"}
            }
        }
    }
    es.indices.create(index=index_name, body=settings)


def insert_batch(batch):
    actions = [
        {
            "_index": "fineweb",
            "_id": _id,
            "_source": {
                "text": text[:5000],
                "url": url,
                "language_score": language_score,
                "token_count": token_count,
            }
        }
        for text, _id, url, language_score, token_count in zip(
            batch["text"],
            batch["id"],
            batch["url"],
            batch["language_score"],
            batch["token_count"],
        )
    ]

    helpers.bulk(es, actions)


def register_repository(repo_name="fineweb_backup", repo_path="/usr/share/elasticsearch/snapshots"):
    body = {
        "type": "fs",
        "settings": {
            "location": fineweb_backup,
            "compress": True
        }
    }
    es.snapshot.create_repository(repository=repo_name, body=body, verify=True)
    print(f"Repository {repo_name} created at {repo_path}")


def create_snapshot(repo_name="fineweb_backup", snapshot_name="snapshot_1"):
    es.snapshot.create(repository=repo_name, snapshot=snapshot_name, wait_for_completion=False)
    print(f"Snapshot {snapshot_name} created in repository {repo_name}")


def main():
    print("Creating index")
    create_index()
    register_repository()

    print("Loading dataset")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        "CC-MAIN-2013-20",
        split="train",
        num_proc=32,
        cache_dir="/scratch/cosmo/.cache",
    )
    dataset = dataset.select_columns(
        ["text", "id", "url", "language_score", "token_count"]
    ).select(range(100000))

    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        insert_batch(batch)
        print(f"Inserted batch {i // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}")

    print("Finished ingestion")
    create_snapshot()


if __name__ == "__main__":
    main()
