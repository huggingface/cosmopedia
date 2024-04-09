import json
from transformers import AutoTokenizer

cats = json.load(open("bisac.json"))
first_cats = sorted(set(f"{cat['top_category']}" for cat in cats))
second_cats = sorted(set(f"{cat['subcategory'].split(' / ')[0]}" for cat in cats if "FICTION" not in cat["top_category"] or "NON-FICTION" in cat["top_category"]))
third_cats = sorted(set(f"{cat['subcategory'].split(' / ')[1]}" for cat in cats if len(cat['subcategory'].split(' / ')) > 1))

cats = sorted(set(f"{cat['top_category']} / {cat['subcategory']}" for cat in cats))

tree = {}
for category in cats:
    levels = category.split(" / ")
    node = tree
    for level in levels:
        if level not in node:
            node[level] = {}
        node = node[level]


def print_tree(node):
    parts = []
    for key, value in node.items():
        if value:
            subtree = print_tree(value)
            part = f"{key} [{subtree}]"
        else:
            part = key
        parts.append(part)
    return '; '.join(parts)


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
tree_repr = print_tree(tree)
print(tree_repr)
# length of the tree representation in tokens
print(len(tokenizer(tree_repr)["input_ids"]))
print(len(tokenizer("; ".join(f'"{cat}"' for cat in first_cats))["input_ids"]))
print(len(tokenizer("; ".join(f'"{cat}"' for cat in (set(second_cats).union(third_cats))))["input_ids"]))
print(len(tokenizer("; ".join(f'"{cat}"' for cat in third_cats))["input_ids"]))

