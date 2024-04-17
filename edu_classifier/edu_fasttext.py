from datasets import load_dataset
import random
import fasttext
import re
from sklearn.linear_model import Ridge

train_dataset = load_dataset(
    "HuggingFaceTB/fineweb_annotated_judge_300k_scores", split="train"
)


# original stats: {1: 13186, 2: 235546, 3: 84285, 4: 9367, 5: 75}
def subsample(row):
    if row["score"] == 2:
        return random.random() < 20000 / 235546
    elif row["score"] == 3:
        return random.random() < 20000 / 84285
    else:
        return True


train_dataset = train_dataset.filter(subsample)

with open("train.txt", "w") as f:
    for text, label in zip(train_dataset["text"], train_dataset["score"]):
        text = " ".join(re.findall(r"\w+", text)).lower()
        f.write(f"__label__{label} {text}\n")


def cleanup(text):
    text = " ".join(re.findall(r"\w+", text)).lower()
    return text


model = fasttext.train_supervised(
    input="train.txt",
    dim=256,
    lr=0.1,
    wordNgrams=3,
    minCount=3,
    minCountLabel=10,
    epoch=3,
)

emb_dataset = train_dataset.map(
    lambda x: {"embeddings": model.get_sentence_vector(cleanup(x["text"]))}
)

ridge = Ridge(alpha=0.01)
ridge.fit(emb_dataset["embeddings"], emb_dataset["score"])


def get_labels(row):
    text = " ".join(re.findall(r"\w+", row["text"])).lower()
    edu_label = int(model.predict(text)[0][0][-1])
    edu_score = ridge.predict([model.get_sentence_vector(text)])[0]
    return {"edu_label": edu_label, "edu_score": edu_score}


pred_data = load_dataset(
    "HuggingFaceFW/FW-12-CC-MAIN-2023-50-15M",
    split="train",
    cache_dir="/scratch/cosmo/cache/",
    num_proc=32,
)

pred_data = pred_data.map(get_labels, num_proc=32)
pred_data.push_to_hub("HuggingFaceTB/FW-2023-50-15M-edu-scores", private=True)
