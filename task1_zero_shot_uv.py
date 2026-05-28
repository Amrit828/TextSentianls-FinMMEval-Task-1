#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "datasets>=3.0.0",
#   "transformers>=4.57.0",
#   "torch",
#   "accelerate>=1.0.0",
# ]
# ///

import json
import os
import re
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Edit these values directly.
# No argparse is used on purpose.
# -----------------------------
MODEL_ID = "google/gemma-3-4b-it"
DATASET_ID = "Tomas08119993/finmmeval-cfa-cpa"
DATASET_CONFIG = None
DATASET_SPLIT = "train"
OUTPUT_PATH = "predictions_task1_zero_shot.jsonl"

# Pick the exact fields for the dataset you want to run.
# Update after checking one sample row if needed.
COLUMN_MAP = {
    "question": "question",
    "options": ["A", "B", "C", "D"],
    "answer": "answer",
    "id": "id",
}

SYSTEM_PROMPT = (
    "You are solving a financial multiple-choice exam question. "
    "Choose the single best answer. "
    "Reply with exactly one capital letter: A, B, C, or D."
)

DTYPE = "auto"  # "auto", "bfloat16", or "float16"
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.0
TOP_P = 1.0
DO_SAMPLE = False
LIMIT = None  # Set an integer like 50 for quick testing


def resolve_dtype(dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return "auto"


def extract_answer(text: str) -> str:
    cleaned = text.strip().upper()

    exact = re.fullmatch(r"[ABCD]", cleaned)
    if exact:
        return cleaned

    for pattern in [
        r"\bANSWER\s*[:\-]?\s*([ABCD])\b",
        r"\bOPTION\s*([ABCD])\b",
        r"\(([ABCD])\)",
        r"\b([ABCD])\b",
    ]:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)

    return "INVALID"


def build_user_prompt(example: dict[str, Any]) -> str:
    question = str(example[COLUMN_MAP["question"]]).strip()
    option_keys = COLUMN_MAP["options"]

    option_lines = []
    for key in option_keys:
        value = example[key]
        option_lines.append(f"{key}. {str(value).strip()}")

    return "\n".join(
        [
            "Question:",
            question,
            "",
            "Options:",
            *option_lines,
            "",
            "Return only one letter: A, B, C, or D.",
        ]
    )


def get_model_and_tokenizer():
    dtype = resolve_dtype(DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, example: dict[str, Any]) -> dict[str, Any]:
    user_prompt = build_user_prompt(example)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pred = extract_answer(raw_text)

    gold = None
    answer_key = COLUMN_MAP.get("answer")
    if answer_key and answer_key in example:
        gold = str(example[answer_key]).strip()

    row_id = None
    id_key = COLUMN_MAP.get("id")
    if id_key and id_key in example:
        row_id = example[id_key]

    result = {
        "id": row_id,
        "prediction": pred,
        "raw_output": raw_text.strip(),
    }
    if gold is not None:
        result["gold"] = gold
        result["correct"] = pred == gold
    return result


def main():
    print(f"Loading dataset: {DATASET_ID} | split={DATASET_SPLIT}")
    if DATASET_CONFIG is None:
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    else:
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)

    if LIMIT is not None:
        dataset = dataset.select(range(min(LIMIT, len(dataset))))

    print(f"Loaded {len(dataset)} rows")
    print("First row keys:", dataset.column_names)
    print("Loading model...")
    model, tokenizer = get_model_and_tokenizer()

    total = 0
    valid = 0
    correct = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for idx, example in enumerate(dataset):
            result = generate_one(model, tokenizer, example)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            total += 1
            if result["prediction"] in {"A", "B", "C", "D"}:
                valid += 1
            if result.get("correct") is True:
                correct += 1

            if (idx + 1) % 10 == 0 or idx == 0:
                msg = (
                    f"[{idx + 1}/{len(dataset)}] "
                    f"pred={result['prediction']} "
                    f"raw={result['raw_output']!r}"
                )
                if "gold" in result:
                    msg += f" gold={result['gold']}"
                print(msg)

    print(f"\nSaved predictions to {OUTPUT_PATH}")
    print(f"Valid answer rate: {valid}/{total} = {valid / max(total, 1):.4f}")
    if total > 0 and correct > 0:
        print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")
    elif total > 0 and "answer" in COLUMN_MAP:
        print("Accuracy: 0.0000")


if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    main()
