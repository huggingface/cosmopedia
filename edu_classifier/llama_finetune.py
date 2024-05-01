import argparse
import multiprocessing
import os

import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
from trl import SFTTrainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceTB/llama3_edu_500k_binary_labels")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--dataset_label_field", type=str, default="binary_target")

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="/fsx/anton/cosmopedia/phi-3-sft-500k")
    parser.add_argument("--cache_dir", type=str, default="/scratch/cosmo/cache")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    return parser.parse_args()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(args):
    # config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # load model and dataset
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    print_trainable_parameters(model)

    data = load_dataset(
        args.dataset_name,
        split=args.split,
        num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
        cache_dir=args.cache_dir,
    )
    data = data.class_encode_column(args.dataset_label_field)
    data = data.train_test_split(
        train_size=0.90, seed=42, stratify_by_column=args.dataset_label_field
    )
    data = data.map(
        lambda x: {"cls_text": f"{x[args.dataset_text_field]}\nScore: {x[args.dataset_label_field]}"},
    )

    # setup the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        max_seq_length=args.max_seq_length,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            logging_strategy="steps",
            logging_steps=10,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            #report_to="wandb",
        ),
        peft_config=lora_config,
        dataset_text_field="cls_text",
    )

    # launch
    print("Training...")
    trainer.train()

    print("Saving the last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    if args.push_to_hub:
        trainer.push_to_hub("Upload model", private=True)
    print("Training Done! 💥")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)