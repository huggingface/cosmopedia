import json
import re
from enum import Enum
from pydantic import BaseModel
from datasets import load_dataset
from text_generation import Client
from text_generation.types import GrammarType, Grammar
from transformers import AutoTokenizer
import argparse
import sys
import time

cats = json.load(open("bisac.json"))
cats = sorted(set(f"{cat['top_category']} / {cat['subcategory']}" for cat in cats))
cats.append("NON-CLASSIFIABLE")
# BisacEnum = Enum("BisacEnum", [(x, x) for x in cats], type=str)
# TextTypeEnum = Enum("TextTypeEnum", [(x, x) for x in ["book", "web"]], type=str)
#
#
# class Labels(BaseModel):
#     bisac_category: BisacEnum
#     text_type: TextTypeEnum

def add_to_tree(base, parts):
    for part in parts:
        if part not in base:
            base[part] = {}
        base = base[part]


def build_regex(node, depth=0):
    if not node:  # If there's no further subdivision
        return ""
    prefix = " \\/ " if depth > 0 else ""  # Add the separator for levels beyond the top
    # Sort keys for consistent ordering, join with '|', and enclose in non-capturing groups
    choices = "|".join(
        sorted(
            f"{re.escape(part)}{build_regex(subnode, depth + 1)}"
            for part, subnode in node.items()
        )
    )
    return f"{prefix}(?:{choices})"


def bisac_regex(categories):
    category_tree = {}
    for category in categories:
        parts = category.split(" / ")
        add_to_tree(category_tree, parts)

    regex = build_regex(category_tree)
    return regex


json_regex = "Based on the provided text, which .{0,256}, I would categorize it as follows:\s{0,16}\{\"\s{0,16}bisac_category\":\s{0,16}\"<BISAC_REGEX>\",\s{0,16}\"text_type\":\s{0,16}\"(?:book|web)\"\s{0,16}\}"
json_regex = json_regex.replace("<BISAC_REGEX>", bisac_regex(cats))


template = """
Your task is to classify a given text using a BISAC category. The text comes from either a book or a web page.

The top-level BISAC categories are: 
ANTIQUES & COLLECTIBLES, ARCHITECTURE, ART, BIBLES, BIOGRAPHY & AUTOBIOGRAPHY, BODY, MIND & SPIRIT, BUSINESS & ECONOMICS, COMICS & GRAPHIC NOVELS, COMPUTERS, COOKING, CRAFTS & HOBBIES, DESIGN, DRAMA, EDUCATION, FAMILY & RELATIONSHIPS, FICTION, FOREIGN LANGUAGE STUDY, GAMES & ACTIVITIES, GARDENING, HEALTH & FITNESS, HISTORY, HOUSE & HOME, HUMOR, JUVENILE FICTION, JUVENILE NONFICTION, LANGUAGE ARTS & DISCIPLINES, LAW, LITERARY COLLECTIONS, LITERARY CRITICISM, MATHEMATICS, MEDICAL, MUSIC, NATURE, PERFORMING ARTS, PETS, PHILOSOPHY, PHOTOGRAPHY, POETRY, POLITICAL SCIENCE, PSYCHOLOGY, REFERENCE, RELIGION, SCIENCE, SELF-HELP, SOCIAL SCIENCE, SPORTS & RECREATION, STUDY AIDS, TECHNOLOGY & ENGINEERING, TRANSPORTATION, TRAVEL, TRUE CRIME, YOUNG ADULT FICTION, YOUNG ADULT NONFICTION

Try to be as specific as possible, rather than assigning a "General" category. If there is no appropriate BISAC category, assign a category "NON-CLASSIFIABLE".
Additionally, determine whether this text comes from a literary or academic work (type "book") or a web page (type "web").
Reply with a valid json object in the following format:
```
{{"bisac_category": "<text category>", "text_type": "<book or web>"}}
```

Text to classify:
```
{text}
```
"""


def main(args):
    dataset = load_dataset(
        "HuggingFaceFW/FW-12-CC-MAIN-2023-50-15M",
        split="train",
        num_proc=16,
        cache_dir="/scratch/cosmo/.cache",
    )

    while True:
        try:
            client = Client(base_url=f"http://localhost:{args.port}", timeout=7200)
            client.generate(prompt="ping", max_new_tokens=1)
            break
        except Exception as e:
            print(e, file=sys.stderr)
            time.sleep(5)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    grammar = Grammar(type=GrammarType.Regex, value=json_regex)
    # cache the grammar FSM object (1h build time) to avoid recompiling it with a smaller timeout
    client.generate(
        "warmup",
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        best_of=2,
        grammar=grammar,
    )

    def classify_text(sample):
        client = Client(base_url=f"http://localhost:{args.port}", timeout=180)
        prompt = template.format(text=sample["text"][:2000])

        response = client.generate(
            prompt=tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
            ),
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            best_of=2,
            grammar=grammar,
        )
        sample["response"] = response.generated_text
        label = json.loads("{" + response.generated_text.split("{")[-1])
        sample["bisac_category"] = label["bisac_category"]
        sample["text_type"] = label["text_type"]
        return sample

    dataset = dataset.select(range(500000)).map(classify_text, num_proc=32)
    dataset.push_to_hub("HuggingFaceTB/FW-2023-50-bisac-classified-explained", private=True)


if __name__ == "__main__":
    # parse the port arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    main(args)
