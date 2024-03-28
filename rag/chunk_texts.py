from multiprocessing import Pool
from transformers import AutoTokenizer
from langchain.text_splitter import MarkdownTextSplitter
import json
from tqdm import tqdm
from smart_open import open


def chunk_text(file_index):
    tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
    markdown_splitter = MarkdownTextSplitter(
        chunk_size=512,
        chunk_overlap=16,
        length_function=lambda x: len(tokenizer(x, max_length=None)["input_ids"]),
    )

    input_path = f"s3://bigcode-datasets-us-east-1/wikipedia/20240301/text/enwiki_namespace_0_{file_index}.jsonl"
    output_path = f"s3://bigcode-datasets-us-east-1/wikipedia/20240301/chunked-mxbai-embed-large-v1/enwiki_{file_index}.jsonl"

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in tqdm(
            fin, position=file_index % 64, desc=f"File {file_index}", leave=False
        ):
            data = json.loads(line)
            _id = data["id"]
            title = data["title"]
            text = data["text"]
            docs = markdown_splitter.create_documents([text])
            for i, doc in enumerate(docs):
                chunked_id = f"{_id}_{i}"
                chunked_text = doc.page_content
                json.dump(
                    {"id": chunked_id, "title": title, "text": chunked_text}, fout
                )
                fout.write("\n")


def main():
    file_indices = range(357)
    num_processes = 64
    with Pool(num_processes) as p:
        p.map(chunk_text, file_indices)


if __name__ == "__main__":
    main()
