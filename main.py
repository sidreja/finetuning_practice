# Code here
# finetune.py
# Requirements: transformers, datasets, accelerate, huggingface_hub
# Optional (recommended for very large models): peft, bitsandbytes
# Usage example:
#   python finetune.py --model_name_or_path "gpt2" --output_dir "./ft-gpt2" --epochs 3 --per_device_train_batch_size 4

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, default=None, help="JSON/JSONL or CSV dataset for training")
    parser.add_argument("--validation_file", type=str, default=None, help="Optional validation file")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true")
    return parser.parse_args()

def load_local_dataset(train_file, validation_file):
    # Accepts json, jsonl, csv; dataset should have a text column named "text" or "input" and "output" for instruction setups.
    # For instruction-finetuning you may have { "instruction": "...", "output": "..." } pairs which are concatenated below.
    data_files = {}
    if train_file:
        data_files["train"] = train_file
    if validation_file:
        data_files["validation"] = validation_file
    if not data_files:
        raise ValueError("Please provide --train_file (and optionally --validation_file).")
    dataset = load_dataset("json", data_files=data_files) if train_file.endswith(".json") or train_file.endswith(".jsonl") else load_dataset("csv", data_files=data_files)
    return dataset

def build_prompt(example: Dict[str, Any]) -> 
