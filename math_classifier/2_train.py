import re
import os
from tqdm.auto import tqdm
from datasets import load_dataset
import fasttext
from sklearn.metrics import classification_report

labeled_fw = load_dataset("HuggingFaceTB/fineweb_math_seeds", split="train", cache_dir="/scratch/anton/.cache/")
os.makedirs('/scratch/anton/fineweb_math_seeds/', exist_ok=True)

def prepare_sample(sample):
    text = ' '.join(re.findall(r'\w+', sample["text"])).lower()
    if sample["label"] == 1:
        return {"ft_sample": f"__label__math {text}"}
    else:
        return {"ft_sample": f"__label__web {text}"}
labeled_fw = labeled_fw.map(prepare_sample, num_proc=60)
labeled_fw = labeled_fw.train_test_split(test_size=0.1, seed=0)

with open('/scratch/anton/fineweb_math_seeds/train.txt', 'w') as ftrain:
    for sample in tqdm(labeled_fw["train"]):
        ftrain.write(sample["ft_sample"] + '\n')


model = fasttext.train_supervised(input='/scratch/anton/fineweb_math_seeds/train.txt', dim=256, lr=0.1, wordNgrams=3, minCount=3, minCountLabel=10, epoch=10, thread=64)
os.makedirs('/fsx/anton/fineweb_math/', exist_ok=True)
model.save_model('/fsx/anton/fineweb_math/math_vs_web.ft')

model = fasttext.load_model('/fsx/anton/fineweb_math/math_vs_web.ft')

pred_labels = [1 if model.predict(sample.split(" ", 1)[1])[0][0] == "__label__math" else 0 for sample in labeled_fw["test"]["ft_sample"]]
true_labels = labeled_fw["test"]["label"]

print(classification_report(true_labels, pred_labels, target_names=['web', 'math'], zero_division=0))


