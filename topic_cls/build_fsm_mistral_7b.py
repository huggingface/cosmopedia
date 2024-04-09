import json
from enum import Enum
from pydantic import BaseModel, field_validator
import cloudpickle
import time
import outlines
from typing import List, Optional


CATEGORIES = json.load(open("bisac.json"))
CATEGORIES = sorted(set(f"{cat['top_category']} / {cat['subcategory']}" for cat in CATEGORIES))
CATEGORIES.append("NON-CLASSIFIABLE")


def generate_regex(categories):
    category_tree = tree()
    for category in categories:
        parts = category.split(" / ")
        add_to_tree(category_tree, parts)

    regex = f"{build_regex(category_tree)}"
    return regex


def build_category_tree(categories):
    tree = {}
    for category in categories:
        levels = category.split(' / ')
        current_level = tree
        for level in levels:
            if level not in current_level:
                current_level[level] = {}
            current_level = current_level[level]
    return tree


CAT_TREE = build_category_tree(CATEGORIES)
print(CAT_TREE)

TextTypeEnum = Enum("TextTypeEnum", [("book", "book"), ("web", "web")], type=str)


class Labels(BaseModel):
    bisac_category: str
    text_type: TextTypeEnum

    # Custom validator for bisac_category
    @field_validator('bisac_category')
    def validate_bisac_category(cls, v, values, **kwargs):
        tree = CAT_TREE

        levels = v.split(" / ")
        for level in levels:
            if level not in tree:
                raise ValueError(f"Category '{v}' does not exist in the given hierarchy.")
            tree = tree[level]
        return v

print(Labels.schema())

model = outlines.models.transformers(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda",
)
t0 = time.time()
generator = outlines.generate.json(model, Labels)
print(f"Created the generator in {time.time() - t0:.2f}s")
cloudpickle.dump(generator.fsm, open("./mistral_generator_fsm.pkl", "wb"))
