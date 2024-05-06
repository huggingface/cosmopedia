import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = load_dataset(args.dataset_name, split="train", cache_dir="/scratch/cosmo/cache/", num_proc=16)

    def compute_scores(batch):
        inputs = tokenizer(batch[args.text_column], return_tensors="pt", padding="longest", truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).float().cpu().numpy()

        batch["score"] = logits.tolist()
        batch["int_score"] = [int(round(max(0, min(score, 5)))) for score in logits]
        return batch

    dataset = dataset.map(compute_scores, batched=True, batch_size=256)
    dataset.push_to_hub(args.output_dataset_name, private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/snowflake_m_edu_reg")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceTB/bisac_3B_generations_college_students")
    parser.add_argument("--output_dataset_name", type=str, default="HuggingFaceTB/bisac_3B_generations_college_students_edu_scores")
    parser.add_argument("--text_column", type=str, default="text")

    args = parser.parse_args()
    main(args)
