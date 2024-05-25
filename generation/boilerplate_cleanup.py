import re
import argparse
from datasets import load_dataset


patterns = [
    # alien stories
    r"^Hello.*?[.!]\s+",
    #r"^I'm( so)? excited to.*?[.!]\s+",
    r"^My name is.*?[.!]\s+",
    r"^You've just arrived.*?[.!]\s+",

    # wikihow
    r"^\*\*Welcome, .*?[.!]\*\*\s+",
    r"^(\*\*)?Warning:.*?[.!]\s+",
    r"^We're thrilled.*?[.!]\s+",

    # middle school
    r"^Welcome, .*?[.!]\s+",
]
patterns = [re.compile(p, flags=re.IGNORECASE|re.MULTILINE) for p in patterns]

def clean_text(sample):
    sample['completion_unfiltered'] = sample['completion']
    for pattern in patterns:
        sample['completion'] = pattern.sub('', sample['completion'].strip())
    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/alien_stories_0_1M_llama3")
    args = parser.parse_args()

    data = load_dataset(args.dataset, split="train", cache_dir="/scratch/cosmo/cache", num_proc=32)
    data = data.map(clean_text, num_proc=32)
    data.push_to_hub(args.dataset, private=True)