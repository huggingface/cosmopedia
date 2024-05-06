from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, ClassLabel
import numpy as np
import evaluate
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix


def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


def main(args):
    dataset = load_dataset(
        args.dataset_name, split="train", cache_dir="/scratch/cosmo/cache/", num_proc=8
    )
    dataset = dataset.map(
        lambda x: {args.target_column: np.clip(int(x[args.target_column]), 0, 5)}, num_proc=8
    )

    dataset = dataset.cast_column(
        args.target_column, ClassLabel(names=[str(i) for i in range(6)])
    )
    dataset = dataset.train_test_split(
        train_size=0.9, seed=42, stratify_by_column=args.target_column
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model_name, num_labels=1, classifier_dropout=0.0, hidden_dropout_prob=0.0)

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        learning_rate=3e-4,
        num_train_epochs=20,
        seed=0,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="Snowflake/snowflake-arctic-embed-m")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceTB/llama3_edu_500k_binary_labels")
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument("--checkpoint_dir", type=str, default="/fsx/anton/cosmopedia/edu_score/bert_snowflake_regression")
    args = parser.parse_args()

    main(args)
