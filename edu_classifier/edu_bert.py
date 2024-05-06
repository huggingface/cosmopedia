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
from sklearn.metrics import classification_report, confusion_matrix


DATASET_NAME = "HuggingFaceTB/llama3_edu_500k_binary_labels"
TARGET_COL = "score"
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m"
CHECKPOINT_DIR = "/fsx/anton/cosmopedia/edu_score/bert_snowflake_regression_4"
EVAL_INTERVAL = 1000


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


def main():
    dataset = load_dataset(
        DATASET_NAME, split="train", cache_dir="/scratch/cosmo/cache/", num_proc=8
    )
    dataset = dataset.map(
        lambda x: {TARGET_COL: np.clip(int(x[TARGET_COL]), 0, 5)}, num_proc=8
    )

    dataset = dataset.cast_column(
        TARGET_COL, ClassLabel(names=[str(i) for i in range(6)])
    )
    dataset = dataset.train_test_split(
        train_size=0.9, seed=42, stratify_by_column=TARGET_COL
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[TARGET_COL])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, classifier_dropout=0.0, hidden_dropout_prob=0.0)

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=EVAL_INTERVAL,
        save_steps=EVAL_INTERVAL,
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

    #trainer.push_to_hub("HuggingFaceTB/bert_reg_edu_500k", private=True)


if __name__ == "__main__":
    main()
