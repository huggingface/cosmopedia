from datasets import load_dataset

ds = load_dataset("teknium/OpenHermes-2.5", split="train", num_proc=36)
drop_sources = ["camelai", "glaive-code-assist"]
drop_categories = ["rp", "gtkm", "coding", "wordgame", "riddle"]

def filter_files(x):
    if x["category"] and x["category"].lower() in drop_categories:
        return False
    if x["source"] and x["source"].lower() in drop_sources:
        return False
    return True

def get_prompt(x):
    conversations = x["conversations"]
    prompt = ""
    for i in range(len(conversations)):
        if conversations[i]["from"] == "human":
            prompt += conversations[i]["value"] + "\n"
            assert conversations[i+1]["from"] == "gpt", f"role is {conversations[i+1]['from']} not 'gpt'!"
            prompt += conversations[i+1]["value"] 
            break
    return {"prompt": prompt}

print("Start...")
print(ds)
print("Language filter...")
ds = ds.filter(lambda x: x["language"] in [None, "English"], num_proc=12)
print(ds)
print("Category & source filter...")
ds_f = ds.filter(filter_files, num_proc=36)
print(ds_f)
ds_f = ds_f.map(get_prompt, num_proc=36)
ds_f = ds_f.remove_columns([col for col in ds_f.column_names if col not in ["prompt", "source", "category"]])
ds_f.push_to_hub("HuggingFaceTB/openhermes_filtered")