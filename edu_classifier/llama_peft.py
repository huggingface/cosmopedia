import evaluate
import numpy as np
import torch
from datasets import ClassLabel, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from accelerate import PartialState

MAX_LEN = 1024
checkpoint = "meta-llama/Meta-Llama-3-8B"
class_weights = [468668/2*425281, 468668/2*43387]

dataset = load_dataset(
    "HuggingFaceTB/llama3_edu_500k_binary_labels", cache_dir="/scratch/cosmo/cache/"
)
dataset = dataset.select_columns(["text", "binary_target"])
dataset = dataset.rename_column(
    "binary_target", "label"
)
dataset = dataset.cast_column("label", ClassLabel(names=['low_quality', 'high_quality']))

dataset = dataset["train"].train_test_split(
    train_size=0.9, seed=42, stratify_by_column="label"
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint,  add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


def roberta_preprocessing_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)


dataset = dataset.map(roberta_preprocessing_function, batched=True)
dataset.set_format("torch")

roberta_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    quantization_config=bnb_config,
    device_map={"": PartialState().process_index},
    num_labels=2,
    offload_folder="/scratch/cosmo/offload",
    trust_remote_code=True,
    cache_dir="/scratch/cosmo/cache/"
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)[
        "precision"
    ]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]

    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "accuracy": accuracy,
    }


class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=model.device, dtype=logits.dtype)
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


model = model.cuda()

lr = 1e-3
batch_size = 1
num_epochs = 10

training_args = TrainingArguments(
    output_dir="/fsx/anton/cosmopedia/llama3-lora-500k",
    learning_rate=lr,
    lr_scheduler_type="constant",
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=8,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1-score",
    # report_to="wandb",
    fp16=False,
    bf16=True,
    gradient_checkpointing=False,
)

trainer = WeightedCELossTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=roberta_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
