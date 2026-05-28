#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "datasets>=3.0.0",
#   "huggingface_hub>=0.32.0",
#   "numpy>=1.26.0",
#   "openai>=1.82.0",
#   "pyarrow>=18.0.0",
#   "scikit-learn>=1.5.0",
# ]
# ///

import json
import os
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from openai import OpenAI
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# Edit these values directly.
# No argparse on purpose.
# -----------------------------
OPENAI_MODEL = "gpt-4o-mini"  # try "o4-mini" too
API_KEY_ENV = "OPENAI_API_KEY"

# Build the retrieval corpus ONLY from public training data.
# Do not include organizer-held dev/final evaluation sets here.
CORPUS_SOURCES = [
    {"dataset_id": "SahmBenchmark/arabic-accounting-mcq_train", "config": None, "split": "train"},
    {"dataset_id": "SahmBenchmark/arabic-accounting-mcq_eval", "config": None, "split": "train"},
    {"dataset_id": "SahmBenchmark/arabic-business-mcq_eval", "config": None, "split": "train"},
    {"dataset_id": "SahmBenchmark/arabic-business-mcq_training_standardized", "config": None, "split": "train"},
    {"dataset_id": "TheFinAI/flare-es-multifin", "config": None, "split": "train"},
    {"dataset_id": "Tomas08119993/finmmeval-cfa-cpa", "config": None, "split": "train"},
    {"dataset_id": "TheFinAI/plutus-multifin", "config": None, "split": "train"},
    # Gated at the time of writing; enable after you accept access conditions:
    # {"dataset_id": "bharatgenai/BhashaBench-Finance", "config": None, "split": "train"},
]

# Run inference on any public split you want to probe locally.
# For official organizer portals, export the questions there and adapt the loader,
# but still keep the retrieval corpus restricted to public training data.
TARGET_SOURCE = {
    "dataset_id": "Tomas08119993/finmmeval-cfa-cpa",
    "config": None,
    "split": "train",
}

OUTPUT_PATH = "task1_rag_gpt4omini_predictions.jsonl"

TOP_K = 5
LIMIT = 25  # set to None for full run
SOURCE_SHEET_FILTERS = ["English", "Chinese", "Hindi"]  # or None for all

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 80


@dataclass
class MCQExample:
    row_id: str
    question: str
    options: dict[str, str]
    answer: str | None
    source: str
    reason: str | None
    raw: dict[str, Any]


def detect_language(dataset_id: str, example: dict[str, Any]) -> str | None:
    source_sheet = normalize_text(example.get("source_sheet"))
    if source_sheet:
        return source_sheet

    lowered_id = dataset_id.lower()
    if "arabic" in lowered_id:
        return "Arabic"
    if "bhashabench" in lowered_id or "hindi" in lowered_id:
        text_blob = normalize_text(example)
        if re.search(r"[\u0900-\u097f]", text_blob):
            return "Hindi"
        return "English"
    if "es-" in lowered_id or "spanish" in lowered_id:
        return "Spanish"
    if "plutus" in lowered_id or "greek" in lowered_id:
        return "Greek"
    if "cfa-cpa" in lowered_id:
        text_blob = normalize_text(example)
        if re.search(r"[\u4e00-\u9fff]", text_blob):
            return "Chinese"
        return "English"
    return None


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def language_allowed(source_sheet: Any) -> bool:
    if not SOURCE_SHEET_FILTERS:
        return True
    normalized = normalize_text(source_sheet).lower()
    allowed = {normalize_text(x).lower() for x in SOURCE_SHEET_FILTERS}
    return normalized in allowed


def find_first_key(example: dict[str, Any], candidates: list[str]) -> str | None:
    lowered = {str(k).lower(): k for k in example.keys()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalize_answer_label(value: Any) -> str | None:
    if value is None:
        return None
    text = normalize_text(value).upper()
    if text in {"A", "B", "C", "D"}:
        return text
    digit_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
    if text in digit_map:
        return digit_map[text]
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return None


def extract_options_from_query(query_text: str) -> dict[str, str] | None:
    text = query_text.replace("\r\n", "\n")
    patterns = [
        r"Options:\s*(.*?)\s*Answer:",
        r"选项:\s*(.*?)\s*Answer:",
        r"Options：\s*(.*?)\s*Answer[:：]",
        r"选项：\s*(.*?)\s*Answer[:：]",
    ]

    block = None
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            block = match.group(1).strip()
            break

    if not block:
        return None

    result = {}
    option_pattern = r"(?im)^\s*([abcd])\.\s*(.+?)\s*$"
    for label, value in re.findall(option_pattern, block):
        result[label.upper()] = normalize_text(value)

    if len(result) == 4 and all(result.values()):
        return result
    return None


def extract_options(example: dict[str, Any]) -> dict[str, str] | None:
    query_key = find_first_key(example, ["query"])
    if query_key is not None:
        parsed = extract_options_from_query(normalize_text(example[query_key]))
        if parsed:
            return parsed

    direct_keys = {}
    for label in ["A", "B", "C", "D"]:
        if label in example:
            direct_keys[label] = normalize_text(example[label])

    if len(direct_keys) == 4 and all(direct_keys.values()):
        return direct_keys

    option_field = find_first_key(
        example,
        ["options", "choices", "candidates", "answers", "option_list"],
    )
    if option_field is None:
        return None

    value = example[option_field]

    if isinstance(value, dict):
        result = {}
        for label in ["A", "B", "C", "D"]:
            if label in value:
                result[label] = normalize_text(value[label])
        if len(result) == 4 and all(result.values()):
            return result

    if isinstance(value, list) and len(value) >= 4:
        return {
            "A": normalize_text(value[0]),
            "B": normalize_text(value[1]),
            "C": normalize_text(value[2]),
            "D": normalize_text(value[3]),
        }

    return None


def answer_from_gold_index(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list) and value:
        value = value[0]
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return None

    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    return mapping.get(idx)


def build_example(
    example: dict[str, Any],
    source_name: str,
    index: int,
    question: str,
    options: dict[str, str],
    answer: str | None,
    reason: str | None,
) -> MCQExample:
    row_id_key = find_first_key(example, ["id", "qid", "question_id", "uid"])
    return MCQExample(
        row_id=str(example[row_id_key]) if row_id_key else f"{source_name}-{index}",
        question=normalize_text(question),
        options=options,
        answer=answer,
        source=source_name,
        reason=normalize_text(reason) if reason else None,
        raw=example,
    )


def parse_finmmeval_cfa_cpa(example: dict[str, Any], source_name: str, index: int) -> MCQExample | None:
    question = normalize_text(example.get("text") or example.get("question"))
    options = extract_options(example)
    if not question or not options:
        return None

    answer = normalize_answer_label(example.get("answer"))
    if answer is None:
        answer = answer_from_gold_index(example.get("gold"))

    return build_example(
        example,
        source_name,
        index,
        question=question,
        options=options,
        answer=answer,
        reason=example.get("reason"),
    )


def parse_generic_mcq(example: dict[str, Any], source_name: str, index: int) -> MCQExample | None:
    question_key = find_first_key(
        example,
        ["text", "question", "query", "prompt", "input", "problem", "stem"],
    )
    if question_key is None:
        return None

    options = extract_options(example)
    if not options:
        return None

    answer_key = find_first_key(
        example,
        ["answer", "label", "correct_answer", "target", "gold"],
    )
    reason_key = find_first_key(example, ["reason", "rationale", "explanation"])

    answer = normalize_answer_label(example.get(answer_key)) if answer_key else None
    if answer is None:
        gold_key = find_first_key(example, ["gold"])
        if gold_key:
            answer = answer_from_gold_index(example.get(gold_key))

    return build_example(
        example,
        source_name,
        index,
        question=example[question_key],
        options=options,
        answer=answer,
        reason=example[reason_key] if reason_key else None,
    )


def canonicalize_example(
    example: dict[str, Any],
    dataset_id: str,
    source_name: str,
    index: int,
) -> MCQExample | None:
    lowered_id = dataset_id.lower()

    if "tomas08119993/finmmeval-cfa-cpa" in lowered_id:
        return parse_finmmeval_cfa_cpa(example, source_name, index)

    return parse_generic_mcq(example, source_name, index)


def iterate_dataset_rows(dataset_id: str, config: str | None, split: str):
    if config is None:
        dataset = load_dataset(dataset_id, split=split)
    else:
        dataset = load_dataset(dataset_id, config, split=split)

    for row in dataset:
        yield dict(row)


def iterate_dataset_rows_streaming(dataset_id: str, config: str | None, split: str):
    if config is None:
        dataset = load_dataset(dataset_id, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_id, config, split=split, streaming=True)

    for row in dataset:
        yield dict(row)


def iterate_dataset_rows_raw_parquet(dataset_id: str, split: str):
    repo_files = list_repo_files(repo_id=dataset_id, repo_type="dataset")
    parquet_files = sorted(
        file_path
        for file_path in repo_files
        if fnmatch(file_path, f"data/{split}-*.parquet") or fnmatch(file_path, f"{split}-*.parquet")
    )

    if not parquet_files:
        raise RuntimeError(f"No parquet files found for split={split} in dataset repo {dataset_id}")

    print(f"Reading {len(parquet_files)} parquet file(s) directly from repo")
    for file_path in parquet_files:
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=file_path)
        table = pq.read_table(local_path)
        rows = table.to_pylist()
        for row in rows:
            yield dict(row)


def prefer_raw_parquet_loader(dataset_id: str) -> bool:
    lowered_id = dataset_id.lower()
    return lowered_id == "tomas08119993/finmmeval-cfa-cpa"


def load_mcq_dataset(dataset_id: str, config: str | None, split: str) -> list[MCQExample]:
    source_name = f"{dataset_id}:{config or '-'}:{split}"
    print(f"Loading {source_name}")

    if prefer_raw_parquet_loader(dataset_id):
        print("Using direct parquet loading for this dataset due to known HF schema mismatch across shards")
        examples = []
        skipped = 0
        row_iter = iterate_dataset_rows_raw_parquet(dataset_id, split)
        for i, row in enumerate(row_iter):
            item = canonicalize_example(dict(row), dataset_id, source_name, i)
            if item is None:
                skipped += 1
                continue
            if not language_allowed(detect_language(dataset_id, item.raw)):
                skipped += 1
                continue
            examples.append(item)

        print(f"Loaded {len(examples)} usable rows from {source_name} (skipped {skipped})")
        return examples

    try:
        print("Using standard dataset loading")
        row_iter = iterate_dataset_rows(dataset_id, config, split)
        examples = []
        skipped = 0
        for i, row in enumerate(row_iter):
            item = canonicalize_example(dict(row), dataset_id, source_name, i)
            if item is None:
                skipped += 1
                continue
            if not language_allowed(detect_language(dataset_id, item.raw)):
                skipped += 1
                continue
            examples.append(item)
    except Exception as exc:
        print(f"Standard dataset loading failed for {source_name}: {exc}")
        try:
            print("Falling back to streaming dataset loading")
            examples = []
            skipped = 0
            row_iter = iterate_dataset_rows_streaming(dataset_id, config, split)
            for i, row in enumerate(row_iter):
                item = canonicalize_example(dict(row), dataset_id, source_name, i)
                if item is None:
                    skipped += 1
                    continue
                if not language_allowed(detect_language(dataset_id, item.raw)):
                    skipped += 1
                    continue
                examples.append(item)
        except Exception as stream_exc:
            print(f"Streaming dataset loading also failed for {source_name}: {stream_exc}")
            print("Falling back to direct parquet loading")
            examples = []
            skipped = 0
            row_iter = iterate_dataset_rows_raw_parquet(dataset_id, split)
            for i, row in enumerate(row_iter):
                item = canonicalize_example(dict(row), dataset_id, source_name, i)
                if item is None:
                    skipped += 1
                    continue
                if not language_allowed(detect_language(dataset_id, item.raw)):
                    skipped += 1
                    continue
                examples.append(item)

    print(f"Loaded {len(examples)} usable rows from {source_name} (skipped {skipped})")
    return examples


def load_corpus_examples() -> list[MCQExample]:
    all_examples = []
    for source in CORPUS_SOURCES:
        try:
            source_examples = load_mcq_dataset(
                source["dataset_id"],
                source.get("config"),
                source["split"],
            )
            all_examples.extend(source_examples)
        except Exception as exc:
            print(f"Skipping corpus source {source['dataset_id']} due to load error: {exc}")
    return all_examples


def load_target_examples() -> list[MCQExample]:
    return load_mcq_dataset(
        TARGET_SOURCE["dataset_id"],
        TARGET_SOURCE.get("config"),
        TARGET_SOURCE["split"],
    )


def example_to_corpus_text(example: MCQExample) -> str:
    lines = [
        example.question,
        f"A. {example.options['A']}",
        f"B. {example.options['B']}",
        f"C. {example.options['C']}",
        f"D. {example.options['D']}",
    ]
    if example.answer:
        lines.append(f"Correct answer: {example.answer}")
    return "\n".join(lines)


def build_retriever(corpus_examples: list[MCQExample]):
    corpus_texts = [example_to_corpus_text(x) for x in corpus_examples]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
    )
    matrix = vectorizer.fit_transform(corpus_texts)
    return vectorizer, matrix


def retrieve_examples(
    query_example: MCQExample,
    corpus_examples: list[MCQExample],
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int,
) -> list[tuple[MCQExample, float]]:
    query_text = example_to_corpus_text(
        MCQExample(
            row_id=query_example.row_id,
            question=query_example.question,
            options=query_example.options,
            answer=None,
            source=query_example.source,
            reason=None,
            raw=query_example.raw,
        )
    )
    query_vector = vectorizer.transform([query_text])
    scores = (query_vector @ matrix.T).toarray()[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        hit = corpus_examples[int(idx)]
        if hit.question == query_example.question and hit.options == query_example.options:
            continue
        results.append((hit, score))
    return results


def build_prompt(query_example: MCQExample, retrieved: list[tuple[MCQExample, float]]) -> str:
    blocks = []
    if retrieved:
        blocks.append("Retrieved public training examples:")
        for rank, (item, score) in enumerate(retrieved, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"Example {rank} (similarity={score:.4f})",
                        f"Question: {item.question}",
                        f"A. {item.options['A']}",
                        f"B. {item.options['B']}",
                        f"C. {item.options['C']}",
                        f"D. {item.options['D']}",
                        f"Correct answer: {item.answer or 'Unknown'}",
                        f"Reason: {item.reason or 'Not provided'}",
                    ]
                )
            )

    blocks.append(
        "\n".join(
            [
                "Target question:",
                query_example.question,
                f"A. {query_example.options['A']}",
                f"B. {query_example.options['B']}",
                f"C. {query_example.options['C']}",
                f"D. {query_example.options['D']}",
                "",
                "Think briefly about the financial reasoning and then choose the single best option.",
                "Return exactly one capital letter: A, B, C, or D.",
            ]
        )
    )
    return "\n\n".join(blocks)


def extract_answer(text: str) -> str:
    cleaned = normalize_text(text).upper()
    if cleaned in {"A", "B", "C", "D"}:
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


def call_openai(client: OpenAI, prompt: str) -> str:
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are a careful financial exam assistant. "
            "Use the retrieved public training examples as weak guidance only. "
            "Do not copy blindly. Solve the target multiple-choice question and return only one capital letter: A, B, C, or D."
        ),
        input=prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.output_text


def main():
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} before running this script.")

    client = OpenAI(api_key=api_key)

    corpus_examples = load_corpus_examples()
    target_examples = load_target_examples()

    if LIMIT is not None:
        target_examples = target_examples[:LIMIT]

    print(f"Testing on {len(target_examples)} question(s)")
    print(f"Building retriever over {len(corpus_examples)} public training examples")
    vectorizer, matrix = build_retriever(corpus_examples)

    total = 0
    valid = 0
    correct = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, query_example in enumerate(target_examples, start=1):
            retrieved = retrieve_examples(
                query_example,
                corpus_examples,
                vectorizer,
                matrix,
                TOP_K,
            )
            prompt = build_prompt(query_example, retrieved)
            raw_output = call_openai(client, prompt)
            prediction = extract_answer(raw_output)

            row = {
                "id": query_example.row_id,
                "prediction": prediction,
                "raw_output": normalize_text(raw_output),
                "question": query_example.question,
                "source": query_example.source,
                "retrieved_ids": [item.row_id for item, _ in retrieved],
            }

            if query_example.answer:
                row["gold"] = query_example.answer
                row["correct"] = prediction == query_example.answer

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            total += 1
            if prediction in {"A", "B", "C", "D"}:
                valid += 1
            if row.get("correct") is True:
                correct += 1

            if i == 1 or i % 10 == 0:
                status = f"[{i}/{len(target_examples)}] pred={prediction} raw={row['raw_output']!r}"
                if "gold" in row:
                    status += f" gold={row['gold']}"
                print(status)

    print(f"\nSaved predictions to {OUTPUT_PATH}")
    print(f"Valid answer rate: {valid}/{total} = {valid / max(total, 1):.4f}")
    if total > 0 and any(example.answer is not None for example in target_examples):
        print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")


if __name__ == "__main__":
    main()
