from huggingface_hub import HfFileSystem
import seaborn as sns
import matplotlib.pyplot as plt
import json
from collections import defaultdict
sns.set_theme()

fs = HfFileSystem()
result_paths = fs.glob("datasets/HuggingFaceTB/eval_results_cosmo2/results/*/*/*.json")
result_paths = sorted(result_paths)

model_gsm8k = defaultdict(float)
for path in result_paths:
    log = json.load(fs.open(path))
    model = log["config_general"]["model_name"]
    model_gsm8k[model] = log["results"]["custom|gsm8k|5"]["qem"]

sorted_models = sorted(model_gsm8k.items(), key=lambda x: x[1], reverse=True)
models, scores = zip(*sorted_models)
plt.figure(figsize=(15, 5))
plt.barh(models, scores)
plt.xlabel("accuracy")
plt.title("Accuracy on 5-shot gsm8k")
plt.tight_layout()
plt.show()

for model, score in sorted_models:
    print(f"* {model}: `{score:.4f}`")


