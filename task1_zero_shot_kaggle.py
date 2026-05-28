#!/usr/bin/env python

import json
import os
import re
import subprocess
import sys
from typing import Any


def ensure_package(package_name: str) -> None:
    try:
        __import__(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])


for pkg in ["datasets", "transformers", "accelerate"]:
    ensure_package(pkg)

try:
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
    import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Edit these values directly.
# Kaggle-friendly, no argparse.
# -----------------------------
MODEL_ID = "google/gemma-2-9b-it"
DATASET_ID = "Tomas08119993/finmmeval-cfa-cpa"
DATASET_CONFIG = None
DATASET_SPLIT = "train"
OUTPUT_PATH = "/kaggle/working/task1_zero_shot_predictions.jsonl"

# Change these to match the exact dataset you pick.
# Common pattern:
#   question -> question text
#   A/B/C/D -> option text columns
#   answer -> gold label column if available
COLUMN_MAP = {
    "question": "question",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "answer": "answer",
    "id": "id",
}

SYSTEM_PROMPT = (
    "You are solving a financial multiple-choice exam question. "
    "Choose the single best answer. "
    "Reply with exactly one capital letter: A, B, C, or D."
)

MAX_NEW_TOKENS = 8
TEMPERATURE = 0.0
TOP_P = 1.0
DO_SAMPLE = False
LIMIT = 50  # set to None for full run
USE_4BIT = False


def extract_answer(text: str) -> str:
    cleaned = text.strip().upper()

    if re.fullmatch(r"[ABCD]", cleaned):
        return cleaned

    for pattern in [
        r"\bANSWER\s*[:\-]?\s*([ABCD])\b",
        r"\(([ABCD])\)",
        r"\b([ABCD])\b",
    ]:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)

    return "INVALID"


def build_user_prompt(example: dict[str, Any]) -> str:
    question = str(example[COLUMN_MAP["question"]]).strip()
    a = str(example[COLUMN_MAP["A"]]).strip()
    b = str(example[COLUMN_MAP["B"]]).strip()
    c = str(example[COLUMN_MAP["C"]]).strip()
    d = str(example[COLUMN_MAP["D"]]).strip()

    return (
        f"Question:\n{question}\n\n"
        f"Options:\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n\n"
        f"Return only one letter: A, B, C, or D."
    )


def get_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": get_dtype(),
    }

    if USE_4BIT:
        ensure_package("bitsandbytes")
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, example: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    prediction = extract_answer(raw_output)

    result = {
        "id": example.get(COLUMN_MAP["id"]) if COLUMN_MAP.get("id") else None,
        "prediction": prediction,
        "raw_output": raw_output,
    }

    answer_key = COLUMN_MAP.get("answer")
    if answer_key and answer_key in example:
        gold = str(example[answer_key]).strip()
        result["gold"] = gold
        result["correct"] = prediction == gold

    return result


def main():
    print("Loading dataset...")
    if DATASET_CONFIG is None:
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    else:
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)

    print("Columns:", dataset.column_names)
    if len(dataset) > 0:
        print("First example:")
        print(dataset[0])

    if LIMIT is not None:
        dataset = dataset.select(range(min(LIMIT, len(dataset))))

    print(f"Running on {len(dataset)} examples")
    print("Loading model...")
    model, tokenizer = get_model_and_tokenizer()

    total = 0
    valid = 0
    correct = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            result = generate_one(model, tokenizer, example)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            total += 1
            if result["prediction"] in {"A", "B", "C", "D"}:
                valid += 1
            if result.get("correct") is True:
                correct += 1

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"[{i + 1}/{len(dataset)}] "
                    f"pred={result['prediction']} "
                    f"raw={result['raw_output']!r}"
                )

    print(f"Saved predictions to: {OUTPUT_PATH}")
    print(f"Valid answer rate: {valid}/{total} = {valid / max(total, 1):.4f}")
    if total > 0 and any("correct" in json.loads(line) for line in open(OUTPUT_PATH, "r", encoding="utf-8")):
        print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")


if __name__ == "__main__":
    main()
