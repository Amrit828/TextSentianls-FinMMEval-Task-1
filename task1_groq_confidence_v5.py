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
# V5 config
# -----------------------------
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

ROUTER_MODEL = "qwen/qwen3-32b"
FIRST_PASS_MODEL = "openai/gpt-oss-20b"
SECOND_PASS_STRONG_MODEL = "openai/gpt-oss-120b"
SECOND_PASS_QWEN_MODEL = "qwen/qwen3-32b"

INDEX_DIR = "task1_faiss_store_groq_v5"
RUN_MODE = "eval_local"  # "build" or "eval_local"
FORCE_REBUILD_INDEX = False

TOP_K_FACTUAL = 2
TOP_K_REASONING = 1
TOP_K_EXHIBIT = 0
LIMIT = 50
PREDICTIONS_PATH = "task1_groq_confidence_v5_predictions.jsonl"

REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 6

MIN_SIMILARITY_FACTUAL = 0.34
MIN_SIMILARITY_REASONING = 0.40
MIN_SIMILARITY_EXHIBIT = 1.01  # effectively disables retrieval


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
            request = dict(kwargs)
            request["timeout"] = REQUEST_TIMEOUT_SECONDS
            return client.chat.completions.create(**request)
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
    canonical_scores: dict[str, Any] = {}

    if isinstance(scores, dict):
        canonical_scores = scores
    elif isinstance(scores, list):
        for idx, value in enumerate(scores):
            if idx >= len(option_labels):
                break
            canonical_scores[option_labels[idx]] = value
    else:
        canonical_scores = {}

    result = {}
    for label in option_labels:
        raw_value = canonical_scores[label] if label in canonical_scores else 0.0
        try:
            result[label] = float(raw_value)
        except (TypeError, ValueError):
            result[label] = 0.0
    total = sum(max(v, 0.0) for v in result.values())
    if total <= 0:
        uniform = 1.0 / max(len(option_labels), 1)
        return {label: uniform for label in option_labels}
    return {label: max(result[label], 0.0) / total for label in option_labels}


def score_margin(scores: dict[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return 1.0
    return ordered[0] - ordered[1]


def best_label(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


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
    data = extract_json_object(response.choices[0].message.content or "{}") or {}
    route = data.get("route", "factual_rag")
    difficulty = data.get("difficulty", "medium")
    if route not in {"factual_rag", "reasoning_math", "exhibit_best_effort"}:
        route = "factual_rag"
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"
    return {"route": route, "difficulty": difficulty, "rationale_short": str(data.get("rationale_short", ""))}


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


def build_option_scoring_prompt(route_info: dict[str, str], example: base.MCQExample, retrieved: list[tuple[base.MCQExample, float]], style: str) -> str:
    route = route_info["route"]
    if route == "reasoning_math":
        route_instruction = "This is likely calculation-heavy. Reason privately and score every option."
    elif route == "exhibit_best_effort":
        route_instruction = "This may reference an exhibit or passage. Infer from visible text and options even if some context is missing."
    else:
        route_instruction = "This is likely factual or conceptual. Use retrieved examples as soft reference patterns."

    style_line = {
        "weak_ranker": "Be concise and prioritize speed.",
        "strong_ranker": "Be careful and prioritize accuracy on difficult questions.",
        "qwen_verifier": "Act as a multilingual verifier of option plausibility.",
    }[style]

    return "\n\n".join(
        [
            "Score each option for correctness.",
            "Return JSON only with keys: scores, best_answer.",
            route_instruction,
            style_line,
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


def model_reasoning_effort(model: str, route_info: dict[str, str], second_pass: bool) -> str | None:
    if not model.startswith("openai/gpt-oss"):
        return None
    if second_pass:
        return "high" if route_info["difficulty"] == "hard" else "medium"
    return "low" if route_info["difficulty"] == "easy" else "medium"


def score_with_model(client: OpenAI, model: str, route_info: dict[str, str], example: base.MCQExample, retrieved: list[tuple[base.MCQExample, float]], style: str, second_pass: bool) -> dict[str, Any]:
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
            {"role": "user", "content": build_option_scoring_prompt(route_info, example, retrieved, style)},
        ],
    }

    effort = model_reasoning_effort(model, route_info, second_pass)
    if effort is not None:
        request_kwargs["reasoning_effort"] = effort

    if model.startswith("openai/gpt-oss"):
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
        best_answer = best_label(scores)
    return {"model": model, "scores": scores, "best_answer": best_answer, "raw_output": content}


def retrieve_for_route(example: base.MCQExample, route_info: dict[str, str], embedder, index, corpus_examples, excluded_target_ids):
    route = route_info["route"]
    if route == "reasoning_math":
        top_k = TOP_K_REASONING
        min_similarity = MIN_SIMILARITY_REASONING
    elif route == "exhibit_best_effort":
        top_k = TOP_K_EXHIBIT
        min_similarity = MIN_SIMILARITY_EXHIBIT
    else:
        top_k = TOP_K_FACTUAL
        min_similarity = MIN_SIMILARITY_FACTUAL

    if top_k <= 0:
        return []

    retrieved = base.retrieve_examples(
        query_example=example,
        embedder=embedder,
        index=index,
        corpus_examples=corpus_examples,
        top_k=top_k,
        excluded_target_ids=excluded_target_ids,
    )
    return [(item, score) for item, score in retrieved if score >= min_similarity]


def pick_route_models(route_info: dict[str, str]) -> tuple[str, str | None]:
    route = route_info["route"]
    if route == "reasoning_math":
        return FIRST_PASS_MODEL, SECOND_PASS_STRONG_MODEL
    if route == "exhibit_best_effort":
        return FIRST_PASS_MODEL, None
    return SECOND_PASS_QWEN_MODEL, FIRST_PASS_MODEL


def combine_two_scores(primary: dict[str, Any], secondary: dict[str, Any] | None, route_info: dict[str, str]) -> tuple[str, dict[str, float]]:
    if secondary is None:
        return primary["best_answer"], dict(primary["scores"])

    labels = list(primary["scores"].keys())
    route = route_info["route"]
    if route == "reasoning_math":
        weights = (0.40, 0.60)
    else:
        weights = (0.60, 0.40)

    merged = {}
    for label in labels:
        merged[label] = weights[0] * primary["scores"][label] + weights[1] * secondary["scores"][label]
    return best_label(merged), merged


def build_index_if_needed():
    base.INDEX_DIR = INDEX_DIR
    base.FORCE_REBUILD_INDEX = FORCE_REBUILD_INDEX
    if RUN_MODE == "build":
        base.build_and_save_index()
        return None, None
    return base.load_index_and_metadata()


def run_eval_local():
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

            primary_model, secondary_model = pick_route_models(route_info)
            primary_style = "qwen_verifier" if primary_model == SECOND_PASS_QWEN_MODEL else "weak_ranker"
            secondary_style = "strong_ranker" if secondary_model == SECOND_PASS_STRONG_MODEL else "weak_ranker"

            primary_result = score_with_model(client, primary_model, route_info, example, retrieved, primary_style, second_pass=False)
            secondary_result = None
            if secondary_model is not None:
                secondary_result = score_with_model(client, secondary_model, route_info, example, retrieved, secondary_style, second_pass=False)

            prediction, final_scores = combine_two_scores(primary_result, secondary_result, route_info)
            first_margin = score_margin(primary_result["scores"])

            row = {
                "id": example.row_id,
                "dataset_id": example.dataset_id,
                "split": example.split,
                "language": example.language,
                "route": route_info["route"],
                "difficulty": route_info["difficulty"],
                "prediction": prediction,
                "scores": final_scores,
                "first_pass_margin": first_margin,
                "used_second_pass": False,
                "retrieved_ids": [item.row_id for item, _ in retrieved],
                "retrieved_datasets": [item.dataset_id for item, _ in retrieved],
                "model_votes": {
                    "primary": primary_result["best_answer"],
                    "secondary": secondary_result["best_answer"] if secondary_result is not None else None,
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
                    f"retrieved={len(retrieved)} primary={primary_model}"
                    + (f" gold={row['gold']}" if "gold" in row else "")
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
