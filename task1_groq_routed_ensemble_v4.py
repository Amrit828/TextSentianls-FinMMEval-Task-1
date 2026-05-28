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
import os
import re
import time
from typing import Any

from openai import APITimeoutError, BadRequestError, OpenAI, RateLimitError

import task1_faiss_rag_openai as base


# -----------------------------
# V4 config
# -----------------------------
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

ROUTER_MODEL = "qwen/qwen3-32b"
FACTUAL_MODEL = "llama-3.3-70b-versatile"
REASONING_MODEL = "openai/gpt-oss-20b"

INDEX_DIR = "task1_faiss_store_groq_v4"
RUN_MODE = "eval_local"  # "build" or "eval_local"
FORCE_REBUILD_INDEX = False

TOP_K_FACTUAL = 2
TOP_K_REASONING = 1
LIMIT = 50
PREDICTIONS_PATH = "task1_groq_routed_ensemble_v4_predictions.jsonl"

REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 6


def get_groq_client() -> OpenAI:
    api_key = os.environ.get(GROQ_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {GROQ_API_KEY_ENV} before running this script.")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def parse_retry_after_seconds(message: str) -> float:
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", message, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 15.0


def groq_chat_create_with_retry(client: OpenAI, **kwargs):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            call_kwargs = dict(kwargs)
            call_kwargs["timeout"] = REQUEST_TIMEOUT_SECONDS
            return client.chat.completions.create(**call_kwargs)
        except RateLimitError as exc:
            last_error = exc
            wait_seconds = parse_retry_after_seconds(str(exc))
            print(f"Rate limited, waiting {wait_seconds:.1f}s before retry {attempt}/{MAX_RETRIES}")
            time.sleep(wait_seconds)
        except APITimeoutError as exc:
            last_error = exc
            wait_seconds = min(5 * attempt, 30)
            print(f"Request timed out, waiting {wait_seconds:.1f}s before retry {attempt}/{MAX_RETRIES}")
            time.sleep(wait_seconds)
    raise last_error


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def normalize_option_scores(scores: dict[str, Any], option_labels: list[str]) -> dict[str, float]:
    result = {label: 0.0 for label in option_labels}
    for label in option_labels:
        try:
            result[label] = float(scores.get(label, 0.0))
        except (TypeError, ValueError):
            result[label] = 0.0
    total = sum(max(v, 0.0) for v in result.values())
    if total <= 0:
        uniform = 1.0 / max(len(option_labels), 1)
        return {label: uniform for label in option_labels}
    return {label: max(result[label], 0.0) / total for label in option_labels}


def build_router_prompt(example: base.MCQExample) -> str:
    return "\n".join(
        [
            "Classify this financial multiple-choice question for routing.",
            'Return JSON only with keys: route, difficulty, rationale_short.',
            'route must be one of: "factual_rag", "reasoning_math", "exhibit_best_effort".',
            'difficulty must be one of: "easy", "medium", "hard".',
            "",
            f"Language: {example.language or 'Unknown'}",
            "Question:",
            example.question,
            "Options:",
            *[f"{k}. {v}" for k, v in example.options.items()],
        ]
    )


def route_question(client: OpenAI, example: base.MCQExample) -> dict[str, str]:
    request_kwargs = {
        "model": ROUTER_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are a routing classifier for financial MCQ solving. Return strict JSON only.",
            },
            {"role": "user", "content": build_router_prompt(example)},
        ],
    }
    try:
        response = groq_chat_create_with_retry(client, **request_kwargs)
    except BadRequestError:
        request_kwargs.pop("response_format", None)
        response = groq_chat_create_with_retry(client, **request_kwargs)
    content = response.choices[0].message.content or "{}"
    data = extract_json_object(content) or {}
    route = data.get("route", "factual_rag")
    difficulty = data.get("difficulty", "medium")
    if route not in {"factual_rag", "reasoning_math", "exhibit_best_effort"}:
        route = "factual_rag"
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"
    return {
        "route": route,
        "difficulty": difficulty,
        "rationale_short": str(data.get("rationale_short", "")),
    }


def build_retrieval_block(retrieved: list[tuple[base.MCQExample, float]]) -> str:
    if not retrieved:
        return "Retrieved examples: none"
    lines = ["Retrieved examples:"]
    for rank, (item, score) in enumerate(retrieved, start=1):
        lines.extend(
            [
                f"Example {rank} (similarity={score:.4f})",
                f"Question: {item.question}",
                *[f"{k}. {v}" for k, v in item.options.items()],
                f"Correct answer: {item.answer or 'Unknown'}",
            ]
        )
    return "\n".join(lines)


def build_scoring_prompt(route_info: dict[str, str], example: base.MCQExample, retrieved: list[tuple[base.MCQExample, float]]) -> str:
    route = route_info["route"]
    if route == "reasoning_math":
        route_instruction = "This is likely calculation-heavy or multi-step. Reason privately and score each option."
    elif route == "exhibit_best_effort":
        route_instruction = "This may reference an exhibit or passage. If some context is missing, infer from visible text and options anyway."
    else:
        route_instruction = "This is likely factual or conceptual. Use retrieved examples as soft reference patterns."

    return "\n\n".join(
        [
            "Score each option for correctness.",
            "Return JSON only with keys: scores, best_answer.",
            route_instruction,
            build_retrieval_block(retrieved),
            "\n".join(
                [
                    f"Target language: {example.language or 'Unknown'}",
                    "Question:",
                    example.question,
                    "Options:",
                    *[f"{k}. {v}" for k, v in example.options.items()],
                ]
            ),
        ]
    )


def score_with_model(client: OpenAI, model: str, route_info: dict[str, str], example: base.MCQExample, retrieved: list[tuple[base.MCQExample, float]]) -> dict[str, Any]:
    option_labels = list(example.options.keys())
    request_kwargs = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful financial multiple-choice solver. "
                    "Reason privately. Return strict JSON only. "
                    f"Allowed answer labels are: {', '.join(option_labels)}."
                ),
            },
            {"role": "user", "content": build_scoring_prompt(route_info, example, retrieved)},
        ],
    }

    if model == REASONING_MODEL:
        request_kwargs["reasoning_effort"] = "medium" if route_info["difficulty"] != "hard" else "high"
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "option_scores",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["scores", "best_answer"],
                    "properties": {
                        "scores": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": option_labels,
                            "properties": {label: {"type": "number"} for label in option_labels},
                        },
                        "best_answer": {"type": "string", "enum": option_labels},
                    },
                },
            },
        }
    else:
        request_kwargs["response_format"] = {"type": "json_object"}

    try:
        response = groq_chat_create_with_retry(client, **request_kwargs)
    except BadRequestError:
        request_kwargs.pop("response_format", None)
        response = groq_chat_create_with_retry(client, **request_kwargs)

    content = response.choices[0].message.content or "{}"
    data = extract_json_object(content) or {}
    scores = normalize_option_scores(data.get("scores", {}), option_labels)
    best_answer = str(data.get("best_answer", "")).strip().upper()
    if best_answer not in option_labels:
        best_answer = max(scores, key=scores.get)
    return {
        "model": model,
        "scores": scores,
        "best_answer": best_answer,
        "raw_output": content,
    }


def ensemble_scores(route_info: dict[str, str], example: base.MCQExample, factual_result: dict[str, Any], reasoning_result: dict[str, Any]) -> tuple[str, dict[str, float]]:
    option_labels = list(example.options.keys())
    route = route_info["route"]

    if route == "reasoning_math":
        weight_factual = 0.30
        weight_reasoning = 0.70
    elif route == "exhibit_best_effort":
        weight_factual = 0.55
        weight_reasoning = 0.45
    else:
        weight_factual = 0.60
        weight_reasoning = 0.40

    merged = {}
    for label in option_labels:
        merged[label] = weight_factual * factual_result["scores"][label] + weight_reasoning * reasoning_result["scores"][label]
    best = max(merged, key=merged.get)
    return best, merged


def retrieve_for_route(example: base.MCQExample, route_info: dict[str, str], embedder, index, corpus_examples, excluded_target_ids):
    top_k = TOP_K_REASONING if route_info["route"] == "reasoning_math" else TOP_K_FACTUAL
    return base.retrieve_examples(
        query_example=example,
        embedder=embedder,
        index=index,
        corpus_examples=corpus_examples,
        top_k=top_k,
        excluded_target_ids=excluded_target_ids,
    )


def build_index_if_needed():
    base.INDEX_DIR = INDEX_DIR
    base.FORCE_REBUILD_INDEX = FORCE_REBUILD_INDEX
    if RUN_MODE == "build":
        base.build_and_save_index()
        return None, None
    return base.load_index_and_metadata()


def run_eval_local() -> None:
    index, corpus_examples = build_index_if_needed()
    if RUN_MODE == "build":
        return

    target_examples = base.load_target_examples()
    target_examples = base.sample_target_examples(target_examples)
    if LIMIT is not None:
        target_examples = target_examples[:LIMIT]
    excluded_target_ids = base.build_excluded_target_ids(target_examples)

    print(f"Testing on {len(target_examples)} question(s)")
    client = get_groq_client()
    embedder = base.get_embedder()

    total = 0
    valid = 0
    correct = 0
    retrieval_hits = 0
    route_counts = {"factual_rag": 0, "reasoning_math": 0, "exhibit_best_effort": 0}

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        for i, example in enumerate(target_examples, start=1):
            route_info = route_question(client, example)
            route_counts[route_info["route"]] += 1

            retrieved = retrieve_for_route(example, route_info, embedder, index, corpus_examples, excluded_target_ids)
            if retrieved:
                retrieval_hits += 1

            factual_result = score_with_model(client, FACTUAL_MODEL, route_info, example, retrieved)
            reasoning_result = score_with_model(client, REASONING_MODEL, route_info, example, retrieved)
            prediction, merged_scores = ensemble_scores(route_info, example, factual_result, reasoning_result)

            row = {
                "id": example.row_id,
                "dataset_id": example.dataset_id,
                "split": example.split,
                "language": example.language,
                "route": route_info["route"],
                "difficulty": route_info["difficulty"],
                "prediction": prediction,
                "scores": merged_scores,
                "retrieved_ids": [item.row_id for item, _ in retrieved],
                "retrieved_datasets": [item.dataset_id for item, _ in retrieved],
                "model_votes": {
                    "factual": factual_result["best_answer"],
                    "reasoning": reasoning_result["best_answer"],
                },
            }
            if example.answer:
                row["gold"] = example.answer
                row["correct"] = prediction == example.answer

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            total += 1
            if prediction in example.options:
                valid += 1
            if row.get("correct") is True:
                correct += 1

            if i == 1 or i % 10 == 0:
                print(
                    f"[{i}/{len(target_examples)}] route={route_info['route']} pred={prediction} "
                    f"retrieved={len(retrieved)}" + (f" gold={row['gold']}" if 'gold' in row else "")
                )

    print(f"\nSaved predictions to {PREDICTIONS_PATH}")
    print(f"Retrieval hits: {retrieval_hits}/{total} = {retrieval_hits / max(total, 1):.4f}")
    print(f"Route counts: {route_counts}")
    print(f"Valid answer rate: {valid}/{total} = {valid / max(total, 1):.4f}")
    if total > 0:
        print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")


def main():
    if RUN_MODE == "build":
        build_index_if_needed()
        return
    if RUN_MODE == "eval_local":
        run_eval_local()
        return
    raise ValueError("RUN_MODE must be 'build' or 'eval_local'")


if __name__ == "__main__":
    main()
