#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "datasets>=3.0.0",
#   "faiss-cpu>=1.8.0",
#   "huggingface_hub>=0.32.0",
#   "numpy>=1.26.0",
#   "openai>=1.82.0",
#   "pyarrow>=18.0.0",
#   "sentence-transformers>=3.0.0",
# ]
# ///

import json
import re

from openai import OpenAI

import task1_faiss_rag_openai as base


# -----------------------------
# V2 settings
# -----------------------------
OPENAI_MODEL = "gpt-4o-mini"
TOP_K = 2
LIMIT = 50
USE_HIDDEN_REASONING = True
PREDICTIONS_PATH = "task1_faiss_rag_predictions_v2.jsonl"


def compact_render_options(options: dict[str, str]) -> list[str]:
    return [f"{label}. {text}" for label, text in options.items()]


def build_prompt_v2(query_example: base.MCQExample, retrieved: list[tuple[base.MCQExample, float]]) -> str:
    blocks = [
        "Solve the target multiple-choice finance question.",
        "Use retrieved examples only as soft reference patterns, not as direct evidence.",
        "If the question mentions an exhibit, table, or passage that is not fully shown, still choose the best answer from the available text and options.",
        "Do not explain. Do not refuse. Do not ask for more information.",
    ]

    if retrieved:
        example_lines = ["Retrieved reference examples:"]
        for rank, (item, _score) in enumerate(retrieved, start=1):
            example_lines.extend(
                [
                    f"Example {rank}",
                    f"Question: {item.question}",
                    *compact_render_options(item.options),
                    f"Answer: {item.answer or 'Unknown'}",
                ]
            )
        blocks.append("\n".join(example_lines))

    target_lines = [
        f"Target language: {query_example.language or 'Unknown'}",
        "Target question:",
        query_example.question,
        *compact_render_options(query_example.options),
        "",
        f"Return exactly one capital letter from: {', '.join(query_example.options.keys())}",
    ]
    blocks.append("\n".join(target_lines))
    return "\n\n".join(blocks)


def extract_answer_v2(text: str, option_labels: list[str]) -> str:
    cleaned = base.normalize_text(text).upper()

    first_char_match = re.match(r"^\s*([A-Z])\b", cleaned)
    if first_char_match and first_char_match.group(1) in option_labels:
        return first_char_match.group(1)

    for pattern in [
        r"\bANSWER\s*[:\-]?\s*([A-Z])\b",
        r"\bOPTION\s*([A-Z])\b",
        r"\(([A-Z])\)",
        r"\b([A-Z])\b",
    ]:
        match = re.search(pattern, cleaned)
        if match and match.group(1) in option_labels:
            return match.group(1)

    return "INVALID"


def call_openai_v2(client: OpenAI, prompt: str, option_labels: list[str]) -> str:
    reasoning_hint = (
        "Reason privately before answering. "
        if USE_HIDDEN_REASONING
        else ""
    )

    request_kwargs = {
        "model": OPENAI_MODEL,
        "instructions": (
            "You are a careful financial exam solver. "
            f"{reasoning_hint}"
            "Always commit to the best available option. "
            "Never say you cannot answer, never mention missing context, and never output explanations. "
            f"Output exactly one capital letter from these options only: {', '.join(option_labels)}."
        ),
        "input": prompt,
        "max_output_tokens": 20,
    }
    if not OPENAI_MODEL.startswith("o"):
        request_kwargs["temperature"] = 0.0

    response = client.responses.create(**request_kwargs)
    return response.output_text


def run_eval_v2() -> None:
    api_key = base.os.environ.get(base.API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {base.API_KEY_ENV} before running evaluation.")

    base.OPENAI_MODEL = OPENAI_MODEL
    base.USE_HIDDEN_REASONING = USE_HIDDEN_REASONING

    index, corpus_examples = base.load_index_and_metadata()
    target_examples = base.load_target_examples()
    if LIMIT is not None:
        target_examples = base.sample_target_examples(target_examples)[:LIMIT]
    excluded_target_ids = base.build_excluded_target_ids(target_examples)

    print(f"Testing on {len(target_examples)} question(s)")
    embedder = base.get_embedder()
    client = OpenAI(api_key=api_key)

    total = 0
    valid = 0
    correct = 0
    retrieved_nonempty = 0

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        for i, query_example in enumerate(target_examples, start=1):
            retrieved = base.retrieve_examples(
                query_example=query_example,
                embedder=embedder,
                index=index,
                corpus_examples=corpus_examples,
                top_k=TOP_K,
                excluded_target_ids=excluded_target_ids,
            )
            if retrieved:
                retrieved_nonempty += 1

            prompt = build_prompt_v2(query_example, retrieved)
            option_labels = list(query_example.options.keys())
            raw_output = call_openai_v2(client, prompt, option_labels)
            prediction = extract_answer_v2(raw_output, option_labels)

            row = {
                "id": query_example.row_id,
                "dataset_id": query_example.dataset_id,
                "split": query_example.split,
                "language": query_example.language,
                "prediction": prediction,
                "raw_output": base.normalize_text(raw_output),
                "question": query_example.question,
                "retrieved_ids": [item.row_id for item, _ in retrieved],
                "retrieved_datasets": [item.dataset_id for item, _ in retrieved],
            }
            if query_example.answer:
                row["gold"] = query_example.answer
                row["correct"] = prediction == query_example.answer

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            total += 1
            if prediction in option_labels:
                valid += 1
            if row.get("correct") is True:
                correct += 1

            if i == 1 or i % 10 == 0:
                print(
                    f"[{i}/{len(target_examples)}] pred={prediction} "
                    f"retrieved={len(retrieved)} raw={row['raw_output']!r}"
                    + (f" gold={row['gold']}" if "gold" in row else "")
                )

    print(f"\nSaved predictions to {PREDICTIONS_PATH}")
    print(f"Questions with retrieval hits: {retrieved_nonempty}/{total} = {retrieved_nonempty / max(total, 1):.4f}")
    print(f"Valid answer rate: {valid}/{total} = {valid / max(total, 1):.4f}")
    if total > 0 and correct >= 0:
        print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")


if __name__ == "__main__":
    run_eval_v2()
