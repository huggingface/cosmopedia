from huggingface_hub import HfFileSystem
import seaborn as sns
import matplotlib.pyplot as plt
import json
from tabulate import tabulate
sns.set_theme()

fs = HfFileSystem()
result_paths = fs.glob("datasets/HuggingFaceTB/eval_results_cosmo2/results/*/*/*.json")
result_paths = sorted(result_paths)

model_gsm8k = {}
for path in result_paths:
    log = json.load(fs.open(path))
    model = log["config_general"]["model_name"]
    #if model.startswith("Hugging"):
    model_gsm8k[model] = (log["results"]["custom|gsm8k|5"]["qem"], log["results"]["custom|gsm8k|5"]["qem_stderr"])

sorted_models = sorted(model_gsm8k.items(), key=lambda x: x[1][0], reverse=True)
models, scores = zip(*sorted_models)
acc, stderr = zip(*scores)
plt.figure(figsize=(15, 8))
plt.barh(models, acc, xerr=stderr, capsize=5)
plt.xlabel("accuracy")
plt.title("Accuracy on 5-shot gsm8k")
plt.tight_layout()
plt.show()

print(tabulate(list(zip(models, acc, stderr))[::-1], ["model", "acc", "stderr"], tablefmt="grid", floatfmt=".4f"))
