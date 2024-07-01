import re
import fasttext
from datasets import load_dataset


dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", cache_dir="/fsx/anton/.cache/", num_proc=60)
model = fasttext.load_model('/fsx/anton/fineweb_math/math_vs_web.ft')

def classify(sample):
    text = ' '.join(re.findall(r'\w+', sample["text"])).lower()
    label, prob = model.predict(text)
    return {"label": int(label[0] == "__label__math"), "prob": prob[0]}


dataset = dataset.map(classify, num_proc=60)
dataset.push_to_hub("HuggingFaceTB/fineweb_10b_math_ft_classified", max_shard_size="2048MB", private=True)