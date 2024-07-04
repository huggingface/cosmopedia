import re
import os
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets
import fasttext
from sklearn.metrics import classification_report

owm_data = load_dataset("HuggingFaceTB/math_classifier_seeds", "owm", split="train", cache_dir="/scratch/anton/.cache/", num_proc=60)
fw_data = load_dataset("HuggingFaceTB/math_classifier_seeds", "fineweb-raw", split="train", cache_dir="/scratch/anton/.cache/", num_proc=60)
os.makedirs('/scratch/anton/fineweb_math_seeds/', exist_ok=True)


def prepare_sample(sample, label):
    text = ' '.join(re.findall(r'\w+', sample["text"])).lower()
    if label == 1:
        return {"ft_sample": f"__label__math {text}", "label": label}
    else:
        return {"ft_sample": f"__label__web {text}", "label": label}


labeled_fw = concatenate_datasets([
    owm_data.map(lambda x: prepare_sample(x, 1), num_proc=60),
    fw_data.map(lambda x: prepare_sample(x, 0), num_proc=60)
])
labeled_fw = labeled_fw.train_test_split(test_size=100_000, seed=0)

with open('/scratch/anton/fineweb_math_seeds/train.txt', 'w') as ftrain:
    for sample in tqdm(labeled_fw["train"]):
        ftrain.write(sample["ft_sample"] + '\n')


model = fasttext.train_supervised(input='/scratch/anton/fineweb_math_seeds/train.txt',
                                  dim=256, lr=0.1, wordNgrams=3, minCount=3, minCountLabel=10, epoch=3, thread=64, seed=0)
os.makedirs('/fsx/anton/fineweb_math/', exist_ok=True)
model.save_model('/fsx/anton/fineweb_math/owm_vs_fw.ft')

model = fasttext.load_model('/fsx/anton/fineweb_math/owm_vs_fw.ft')

pred_labels = [1 if model.predict(sample.split(" ", 1)[1])[0][0] == "__label__math" else 0 for sample in labeled_fw["test"]["ft_sample"]]
true_labels = labeled_fw["test"]["label"]

print(classification_report(true_labels, pred_labels, target_names=['web', 'math'], zero_division=0))


