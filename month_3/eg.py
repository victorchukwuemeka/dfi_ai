"""
LoRA Fine-Tuning Script for Causal Language Models
====================================================

Fine-tunes a pre-trained causal LM on instruction-style data using
LoRA (Low-Rank Adaptation). Only adapter weights are updated (~0.3%
of total parameters), producing a tiny checkpoint file (~MBs).

Usage:
    python eg.py

Requirements:
    torch, transformers, datasets, peft, accelerate
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def load_model_and_tokenizer(
    model_name: str = "microsoft/phi-2",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a pre-trained causal LM and its tokenizer.

    The pad token is set to the EOS token because many models
    do not ship with a native pad token, which is required for
    batched training.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def apply_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Wrap the model with LoRA adapters on the Q and V projections.

    Only ~0.3% of parameters become trainable, keeping memory and
    storage requirements low.
    """
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def prepare_dataset(
    raw_data: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> Dataset:
    """Convert raw instruction-response pairs into a tokenized Dataset.

    Each example is formatted as:
        Instruction: {input}
        Response: {output}

    Then tokenized with truncation and padding to *max_length*.
    """
    def format_example(example: dict[str, str]) -> dict[str, str]:
        return {
            "text": f"Instruction: {example['input']}\nResponse: {example['output']}"
        }

    formatted = [format_example(ex) for ex in raw_data]
    dataset = Dataset.from_list(formatted)

    def tokenize_function(examples: dict[str, list]) -> dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(tokenize_function, batched=True)


def train() -> None:
    """Run the full LoRA fine-tuning pipeline."""
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)

    train_data = [
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris.",
        },
        {
            "input": "Explain gravity simply.",
            "output": (
                "Gravity is a force that pulls objects with mass toward "
                "each other. It's why apples fall from trees and why we "
                "stay on the ground."
            ),
        },
    ]

    tokenized_dataset = prepare_dataset(train_data, tokenizer)

    training_args = TrainingArguments(
        output_dir="./lora-adapters",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    model.save_pretrained("./lora-adapters/final")
    tokenizer.save_pretrained("./lora-adapters/final")
    print("Training complete! Adapter saved to ./lora-adapters/final")


if __name__ == "__main__":
    train()
