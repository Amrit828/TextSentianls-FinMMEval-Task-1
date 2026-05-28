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
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
import random
from typing import Any
from collections import Counter, defaultdict

import faiss
import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -----------------------------
# Edit these values directly.
# No argparse on purpose.
# -----------------------------
OPENAI_MODEL = "gpt-4o-mini"  # or "o4-mini"
API_KEY_ENV = "OPENAI_API_KEY"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_DIR = "task1_faiss_store"
RUN_MODE = "eval"
FORCE_REBUILD_INDEX = False

CORPUS_SOURCES = [
    {"dataset_id": "SahmBenchmark/arabic-accounting-mcq_train", "config": None, "split": "train"},
    {"dataset_id": "SahmBenchmark/arabic-business-mcq_training_standardized", "config": None, "split": "train"},
    {"dataset_id": "TheFinAI/flare-es-multifin", "config": None, "split": "test"},
    {"dataset_id": "Tomas08119993/finmmeval-cfa-cpa", "config": None, "split": "train"},
    {"dataset_id": "TheFinAI/plutus-multifin", "config": None, "split": "train"},
]

TARGET_SOURCE = None
TARGET_SOURCES = [
    {"dataset_id": "SahmBenchmark/arabic-accounting-mcq_eval", "config": None, "split": "test"},
    {"dataset_id": "SahmBenchmark/arabic-business-mcq_eval", "config": None, "split": "test"},
    {"dataset_id": "TheFinAI/plutus-multifin", "config": None, "split": "validation"},
    {"dataset_id": "Tomas08119993/finmmeval-cfa-cpa", "config": None, "split": "train"},
]

TARGET_LANGUAGE_FILTERS = ["English", "Chinese", "Arabic", "Greek"]  # None for all
CORPUS_LANGUAGE_FILTERS = ["English", "Chinese", "Hindi", "Arabic", "Spanish", "Greek"]  # None for all

TOP_K = 3
LIMIT = 50  # None for full target split
EXCLUDE_SELF_MATCH = True
EXCLUDE_TARGET_DATASET_FROM_RETRIEVAL = False
RANDOM_SEED = 42

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 80
PREDICTIONS_PATH = "task1_faiss_rag_predictions.jsonl"
SKIP_DEBUG_PATH = "task1_schema_debug.jsonl"
MAX_SKIP_DEBUG_EXAMPLES_PER_DATASET = 5
USE_HIDDEN_REASONING = True
HIDDEN_REASONING_HINT = "Reason through the retrieved evidence privately before answering."


@dataclass
class MCQExample:
    row_id: str
    dataset_id: str
    split: str
    language: str | None
    question: str
    options: dict[str, str]
    answer: str | None
    reason: str | None
    query: str | None
    raw: dict[str, Any]


LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_language_filters(filters: list[str] | None) -> set[str] | None:
    if not filters:
        return None
    return {normalize_text(x).lower() for x in filters}


def language_allowed(language: str | None, allowed_filters: list[str] | None) -> bool:
    normalized = normalize_language_filters(allowed_filters)
    if normalized is None:
        return True
    language_norm = normalize_text(language).lower()
    alias_map = {
        "en": "english",
        "english": "english",
        "zh": "chinese",
        "chinese": "chinese",
        "hi": "hindi",
        "hindi": "hindi",
        "ar": "arabic",
        "arabic": "arabic",
        "es": "spanish",
        "spanish": "spanish",
        "el": "greek",
        "greek": "greek",
    }
    return alias_map.get(language_norm, language_norm) in normalized


def detect_language(dataset_id: str, example: dict[str, Any]) -> str | None:
    language_field = normalize_text(example.get("language")).lower()
    if language_field == "en":
        return "English"
    if language_field == "hi":
        return "Hindi"
    if language_field == "ar":
        return "Arabic"
    if language_field == "es":
        return "Spanish"
    if language_field == "el":
        return "Greek"
    if language_field == "zh":
        return "Chinese"
    if language_field == "greek":
        return "Greek"
    if language_field == "english":
        return "English"
    if language_field == "hindi":
        return "Hindi"
    if language_field == "arabic":
        return "Arabic"
    if language_field == "spanish":
        return "Spanish"
    if language_field == "chinese":
        return "Chinese"

    source_sheet = normalize_text(example.get("source_sheet"))
    if source_sheet:
        return source_sheet

    text_blob = normalize_text(example.get("query") or example.get("text") or example.get("question") or example)
    lowered_id = dataset_id.lower()

    if "arabic" in lowered_id or re.search(r"[\u0600-\u06ff]", text_blob):
        return "Arabic"
    if "bhashabench" in lowered_id or re.search(r"[\u0900-\u097f]", text_blob):
        return "Hindi"
    if "spanish" in lowered_id or "flare-es" in lowered_id:
        return "Spanish"
    if "greek" in lowered_id or "plutus" in lowered_id or re.search(r"[\u0370-\u03ff]", text_blob):
        return "Greek"
    if "cfa-cpa" in lowered_id:
        if re.search(r"[\u4e00-\u9fff]", text_blob):
            return "Chinese"
        return "English"
    return None


def find_first_key(example: dict[str, Any], candidates: list[str]) -> str | None:
    lowered = {str(k).lower(): k for k in example.keys()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def answer_from_index(value: Any, option_count: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, list) and value:
        value = value[0]
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < option_count:
        return LABELS[idx]
    return None


def normalize_answer_label(value: Any, option_labels: list[str]) -> str | None:
    if value is None:
        return None
    text = normalize_text(value).upper()

    if text in option_labels:
        return text

    lower_option_map = {label.lower(): label for label in option_labels}
    if text.lower() in lower_option_map:
        return lower_option_map[text.lower()]

    digit_map = {str(i + 1): option_labels[i] for i in range(len(option_labels))}
    if text in digit_map:
        return digit_map[text]

    match = re.search(r"\b([A-Z])\b", text)
    if match and match.group(1) in option_labels:
        return match.group(1)

    return None


def parse_options_from_query(query_text: str) -> dict[str, str] | None:
    text = query_text.replace("\r\n", "\n")
    block = text

    block_match = re.search(
        r"(?is)(?:options|choices|option|choice|选项|خيارات|الخيارات|é€‰é¡¹)\s*[:：]?\s*(.*?)(?:answer|correct answer|答案|الإجابة|ç­”|å›žç­”|$)",
        text,
    )
    if block_match:
        block = block_match.group(1).strip()

    result = {}
    regex = re.compile(r"(?is)([a-d])[\.\)]\s*(.*?)(?=(?:\s+[a-d][\.\)]\s)|$)")
    for label, value in regex.findall(block):
        cleaned = normalize_text(value).rstrip(";:,")
        result[label.upper()] = cleaned

    if len(result) >= 2:
        return {label: result[label] for label in LABELS if label in result}
    return None


def parse_options_from_choices(raw_choices: Any) -> dict[str, str] | None:
    if isinstance(raw_choices, dict):
        result = {}
        for key, value in raw_choices.items():
            label = normalize_text(key).upper()
            if len(label) == 1 and label in LABELS:
                result[label] = normalize_text(value)
        if len(result) >= 2:
            return {label: result[label] for label in sorted(result)}

    if isinstance(raw_choices, list) and raw_choices:
        normalized_choices = [normalize_text(x) for x in raw_choices]
        if all(choice for choice in normalized_choices):
            single_letters = all(re.fullmatch(r"[a-zA-Z]", choice) for choice in normalized_choices)
            if single_letters:
                return None
            return {
                LABELS[i]: choice
                for i, choice in enumerate(normalized_choices)
                if i < len(LABELS)
            }
    return None


def extract_options(example: dict[str, Any]) -> dict[str, str] | None:
    query_key = find_first_key(example, ["query"])
    if query_key:
        parsed = parse_options_from_query(normalize_text(example[query_key]))
        if parsed:
            return parsed

    direct = {}
    for label in LABELS[:6]:
        if label in example:
            direct[label] = normalize_text(example[label])
    if len(direct) >= 2 and all(direct.values()):
        return direct

    option_field_map = {}
    for label in ["A", "B", "C", "D", "E", "F"]:
        for variant in [f"option_{label.lower()}", f"option_{label}", f"choice_{label.lower()}", f"choice_{label}"]:
            if variant in example:
                option_field_map[label] = normalize_text(example[variant])
                break
    if len(option_field_map) >= 2 and all(option_field_map.values()):
        return option_field_map

    choices_key = find_first_key(example, ["choices", "options", "answers", "candidates", "option_list"])
    if choices_key:
        parsed = parse_options_from_choices(example[choices_key])
        if parsed:
            return parsed

    return None


def build_example(
    example: dict[str, Any],
    dataset_id: str,
    split: str,
    index: int,
    question: str,
    options: dict[str, str],
    answer: str | None,
    reason: str | None,
) -> MCQExample:
    row_id_key = find_first_key(example, ["id", "qid", "question_id", "uid"])
    query_key = find_first_key(example, ["query"])
    return MCQExample(
        row_id=str(example[row_id_key]) if row_id_key else f"{dataset_id}:{split}:{index}",
        dataset_id=dataset_id,
        split=split,
        language=detect_language(dataset_id, example),
        question=normalize_text(question),
        options=options,
        answer=answer,
        reason=normalize_text(reason) if reason else None,
        query=normalize_text(example.get(query_key)) if query_key else None,
        raw=example,
    )


def parse_finmmeval_cfa_cpa(example: dict[str, Any], dataset_id: str, split: str, index: int) -> MCQExample | None:
    question = normalize_text(example.get("text") or example.get("question"))
    options = extract_options(example)
    if not question or not options:
        return None

    option_labels = list(options.keys())
    answer = normalize_answer_label(example.get("answer"), option_labels)
    if answer is None:
        answer = answer_from_index(example.get("gold"), len(option_labels))

    return build_example(
        example,
        dataset_id,
        split,
        index,
        question=question,
        options=options,
        answer=answer,
        reason=example.get("reason"),
    )


def parse_generic_mcq(example: dict[str, Any], dataset_id: str, split: str, index: int) -> MCQExample | None:
    question_key = find_first_key(example, ["text", "question", "prompt", "input", "problem", "stem", "query"])
    if question_key is None:
        return None

    options = extract_options(example)
    if not options:
        return None

    option_labels = list(options.keys())
    answer_key = find_first_key(example, ["answer", "label", "correct_answer", "target", "gold"])
    reason_key = find_first_key(example, ["reason", "rationale", "explanation"])

    answer = normalize_answer_label(example.get(answer_key), option_labels) if answer_key else None
    if answer is None:
        gold_key = find_first_key(example, ["gold"])
        if gold_key:
            answer = answer_from_index(example.get(gold_key), len(option_labels))

    return build_example(
        example,
        dataset_id,
        split,
        index,
        question=example[question_key],
        options=options,
        answer=answer,
        reason=example[reason_key] if reason_key else None,
    )


def canonicalize_example(example: dict[str, Any], dataset_id: str, split: str, index: int) -> MCQExample | None:
    lowered_id = dataset_id.lower()
    if lowered_id == "tomas08119993/finmmeval-cfa-cpa":
        return parse_finmmeval_cfa_cpa(example, dataset_id, split, index)
    return parse_generic_mcq(example, dataset_id, split, index)


def classify_skip_reason(example: dict[str, Any]) -> str:
    question_key = find_first_key(example, ["text", "question", "prompt", "input", "problem", "stem", "query"])
    if question_key is None:
        return "missing_question_field"

    question_text = normalize_text(example.get(question_key))
    if not question_text:
        return "empty_question"

    options = extract_options(example)
    if not options:
        choices_key = find_first_key(example, ["choices", "options", "answers", "candidates", "option_list"])
        if choices_key is not None:
            raw_choices = example.get(choices_key)
            return f"unparsed_choices_type:{type(raw_choices).__name__}"
        return "missing_or_unparsed_options"

    return "unknown_parser_rejection"


def iterate_dataset_rows_standard(dataset_id: str, config: str | None, split: str):
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
        for row in table.to_pylist():
            yield dict(row)


def prefer_raw_parquet_loader(dataset_id: str) -> bool:
    return dataset_id.lower() == "tomas08119993/finmmeval-cfa-cpa"


def load_mcq_dataset(
    dataset_id: str,
    config: str | None,
    split: str,
    language_filters: list[str] | None,
) -> list[MCQExample]:
    source_name = f"{dataset_id}:{config or '-'}:{split}"
    print(f"Loading {source_name}")
    skip_reason_counts = Counter()
    skip_debug_examples = []

    strategies = []
    if prefer_raw_parquet_loader(dataset_id):
        strategies = [("direct parquet", iterate_dataset_rows_raw_parquet(dataset_id, split))]
    else:
        strategies = [
            ("standard", iterate_dataset_rows_standard(dataset_id, config, split)),
            ("streaming", iterate_dataset_rows_streaming(dataset_id, config, split)),
            ("direct parquet", iterate_dataset_rows_raw_parquet(dataset_id, split)),
        ]

    last_error = None
    for strategy_name, row_iter in strategies:
        try:
            print(f"Using {strategy_name} loading")
            examples = []
            skipped = 0
            for i, row in enumerate(row_iter):
                item = canonicalize_example(dict(row), dataset_id, split, i)
                if item is None:
                    skipped += 1
                    reason = classify_skip_reason(dict(row))
                    skip_reason_counts[reason] += 1
                    if len(skip_debug_examples) < MAX_SKIP_DEBUG_EXAMPLES_PER_DATASET:
                        skip_debug_examples.append(
                            {
                                "dataset": source_name,
                                "reason": reason,
                                "row_index": i,
                                "keys": sorted(list(dict(row).keys())),
                                "row": dict(row),
                            }
                        )
                    continue
                if not language_allowed(item.language, language_filters):
                    skipped += 1
                    skip_reason_counts[f"filtered_language:{item.language or 'Unknown'}"] += 1
                    continue
                examples.append(item)
            print(f"Loaded {len(examples)} usable rows from {source_name} (skipped {skipped})")
            if skip_reason_counts:
                print(f"Skip reasons for {source_name}: {dict(skip_reason_counts)}")
            if skip_debug_examples:
                with open(SKIP_DEBUG_PATH, "a", encoding="utf-8") as debug_f:
                    for record in skip_debug_examples:
                        debug_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return examples
        except Exception as exc:
            last_error = exc
            print(f"{strategy_name.capitalize()} loading failed for {source_name}: {exc}")

    raise RuntimeError(f"Could not load {source_name}") from last_error


def load_corpus_examples() -> list[MCQExample]:
    all_examples = []
    for source in CORPUS_SOURCES:
        try:
            source_examples = load_mcq_dataset(
                source["dataset_id"],
                source.get("config"),
                source["split"],
                CORPUS_LANGUAGE_FILTERS,
            )
            all_examples.extend(source_examples)
        except Exception as exc:
            print(f"Skipping corpus source {source['dataset_id']} due to load error: {exc}")
    return all_examples


def load_target_examples() -> list[MCQExample]:
    if TARGET_SOURCE is not None:
        return load_mcq_dataset(
            TARGET_SOURCE["dataset_id"],
            TARGET_SOURCE.get("config"),
            TARGET_SOURCE["split"],
            TARGET_LANGUAGE_FILTERS,
        )

    combined = []
    for source in TARGET_SOURCES:
        try:
            source_examples = load_mcq_dataset(
                source["dataset_id"],
                source.get("config"),
                source["split"],
                TARGET_LANGUAGE_FILTERS,
            )
            combined.extend(source_examples)
        except Exception as exc:
            print(f"Skipping target source {source['dataset_id']} due to load error: {exc}")
    return combined


def sample_target_examples(target_examples: list[MCQExample]) -> list[MCQExample]:
    if LIMIT is None or LIMIT >= len(target_examples):
        return target_examples

    rng = random.Random(RANDOM_SEED)
    sampled = list(target_examples)
    rng.shuffle(sampled)
    return sampled[:LIMIT]


def build_excluded_target_ids(target_examples: list[MCQExample]) -> set[tuple[str, str]]:
    return {(example.dataset_id, example.row_id) for example in target_examples}


def load_target_examples_deprecated() -> list[MCQExample]:
    return load_mcq_dataset(
        TARGET_SOURCE["dataset_id"],
        TARGET_SOURCE.get("config"),
        TARGET_SOURCE["split"],
        TARGET_LANGUAGE_FILTERS,
    )


def render_options(options: dict[str, str]) -> list[str]:
    return [f"{label}. {text}" for label, text in options.items()]


def example_to_passage(example: MCQExample) -> str:
    lines = [
        f"Dataset: {example.dataset_id}",
        f"Language: {example.language or 'Unknown'}",
        "Question:",
        example.question,
        "Options:",
        *render_options(example.options),
    ]
    if example.answer:
        lines.append(f"Correct answer: {example.answer}")
    if example.reason:
        lines.append(f"Reason: {example.reason}")
    return "\n".join(lines)


def example_to_query(example: MCQExample) -> str:
    lines = [
        f"Language: {example.language or 'Unknown'}",
        "Question:",
        example.question,
        "Options:",
        *render_options(example.options),
    ]
    return "\n".join(lines)


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def encode_texts(embedder: SentenceTransformer, texts: list[str], prompt_type: str) -> np.ndarray:
    prefix = "passage: " if prompt_type == "passage" else "query: "
    encoded = embedder.encode(
        [prefix + text for text in texts],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return encoded.astype("float32")


def save_metadata(metadata_path: Path, examples: list[MCQExample]) -> None:
    with metadata_path.open("w", encoding="utf-8") as f:
        for example in examples:
            row = asdict(example)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_metadata(metadata_path: Path) -> list[MCQExample]:
    examples = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(MCQExample(**row))
    return examples


def build_and_save_index() -> tuple[faiss.Index, list[MCQExample]]:
    index_dir = Path(INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)

    corpus_examples = load_corpus_examples()
    if not corpus_examples:
        raise RuntimeError("No usable corpus examples were loaded.")

    texts = [example_to_passage(x) for x in corpus_examples]
    print(f"Encoding {len(texts)} corpus examples with {EMBED_MODEL}")
    embedder = get_embedder()
    embeddings = encode_texts(embedder, texts, "passage")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "task1.index"))
    save_metadata(index_dir / "task1_metadata.jsonl", corpus_examples)

    manifest = {
        "embed_model": EMBED_MODEL,
        "corpus_sources": CORPUS_SOURCES,
        "corpus_language_filters": CORPUS_LANGUAGE_FILTERS,
        "index_size": len(corpus_examples),
        "dimension": int(embeddings.shape[1]),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved FAISS index and metadata to {index_dir.resolve()}")
    return index, corpus_examples


def load_index_and_metadata() -> tuple[faiss.Index, list[MCQExample]]:
    index_dir = Path(INDEX_DIR)
    index_path = index_dir / "task1.index"
    metadata_path = index_dir / "task1_metadata.jsonl"

    if FORCE_REBUILD_INDEX or not index_path.exists() or not metadata_path.exists():
        print("Index files are missing or rebuild was requested")
        return build_and_save_index()

    index = faiss.read_index(str(index_path))
    metadata = load_metadata(metadata_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors from {index_dir.resolve()}")
    return index, metadata


def retrieve_examples(
    query_example: MCQExample,
    embedder: SentenceTransformer,
    index: faiss.Index,
    corpus_examples: list[MCQExample],
    top_k: int,
    excluded_target_ids: set[tuple[str, str]],
) -> list[tuple[MCQExample, float]]:
    query_vector = encode_texts(embedder, [example_to_query(query_example)], "query")
    scores, indices = index.search(query_vector, min(top_k + 10, max(top_k, 1) + 10))

    results = []
    for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
        if idx < 0:
            continue
        item = corpus_examples[idx]
        if (item.dataset_id, item.row_id) in excluded_target_ids:
            continue
        if EXCLUDE_TARGET_DATASET_FROM_RETRIEVAL and item.dataset_id == query_example.dataset_id:
            continue
        if EXCLUDE_SELF_MATCH and item.dataset_id == query_example.dataset_id and item.row_id == query_example.row_id:
            continue
        if TARGET_LANGUAGE_FILTERS and item.language and query_example.language and item.language != query_example.language:
            continue
        results.append((item, float(score)))
        if len(results) >= top_k:
            break
    return results


def build_prompt(query_example: MCQExample, retrieved: list[tuple[MCQExample, float]]) -> str:
    blocks = []
    if retrieved:
        blocks.append("Retrieved public training examples:")
        for rank, (item, score) in enumerate(retrieved, start=1):
            block = [
                f"Example {rank} (similarity={score:.4f})",
                f"Language: {item.language or 'Unknown'}",
                f"Question: {item.question}",
                *render_options(item.options),
                f"Correct answer: {item.answer or 'Unknown'}",
            ]
            if item.reason:
                block.append(f"Reason: {item.reason}")
            blocks.append("\n".join(block))

    blocks.append(
        "\n".join(
            [
                f"Target language: {query_example.language or 'Unknown'}",
                "Target question:",
                query_example.question,
                *render_options(query_example.options),
                "",
                "Choose the single best option for the target question.",
                "Return exactly one capital letter from the available options only.",
            ]
        )
    )
    return "\n\n".join(blocks)


def extract_answer(text: str, option_labels: list[str]) -> str:
    cleaned = normalize_text(text).upper()
    if cleaned in option_labels:
        return cleaned
    for pattern in [
        r"\bANSWER\s*[:\-]?\s*([A-Z])\b",
        r"\(([A-Z])\)",
        r"\b([A-Z])\b",
    ]:
        match = re.search(pattern, cleaned)
        if match and match.group(1) in option_labels:
            return match.group(1)
    return "INVALID"


def call_openai(client: OpenAI, prompt: str, option_labels: list[str]) -> str:
    reasoning_hint = HIDDEN_REASONING_HINT if USE_HIDDEN_REASONING else ""
    request_kwargs = {
        "model": OPENAI_MODEL,
        "instructions": (
            "You are a careful financial multiple-choice assistant. "
            "Use the retrieved public training examples as guidance only. "
            f"{reasoning_hint} "
            f"Return exactly one capital letter from these options: {', '.join(option_labels)}."
        ),
        "input": prompt,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if not OPENAI_MODEL.startswith("o"):
        request_kwargs["temperature"] = TEMPERATURE

    response = client.responses.create(**request_kwargs)
    return response.output_text


def run_eval() -> None:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} before running evaluation.")

    index, corpus_examples = load_index_and_metadata()
    target_examples = load_target_examples()
    target_examples = sample_target_examples(target_examples)
    excluded_target_ids = build_excluded_target_ids(target_examples)

    print(f"Testing on {len(target_examples)} question(s)")
    embedder = get_embedder()
    client = OpenAI(api_key=api_key)

    total = 0
    valid = 0
    correct = 0

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        for i, query_example in enumerate(target_examples, start=1):
            retrieved = retrieve_examples(
                query_example,
                embedder,
                index,
                corpus_examples,
                TOP_K,
                excluded_target_ids,
            )
            prompt = build_prompt(query_example, retrieved)
            option_labels = list(query_example.options.keys())
            raw_output = call_openai(client, prompt, option_labels)
            prediction = extract_answer(raw_output, option_labels)

            row = {
                "id": query_example.row_id,
                "dataset_id": query_example.dataset_id,
                "split": query_example.split,
                "language": query_example.language,
                "prediction": prediction,
                "raw_output": normalize_text(raw_output),
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
                status = f"[{i}/{len(target_examples)}] pred={prediction} raw={row['raw_output']!r}"
                if "gold" in row:
                    status += f" gold={row['gold']}"
                print(status)

    print(f"\nSaved predictions to {PREDICTIONS_PATH}")
    print(f"Valid answer rate: {valid}/{total} = {valid / max(total, 1):.4f}")
    if total > 0 and any(example.answer is not None for example in target_examples):
        print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")


def main() -> None:
    if RUN_MODE == "build":
        build_and_save_index()
        return
    if RUN_MODE == "eval":
        run_eval()
        return
    raise ValueError("RUN_MODE must be either 'build' or 'eval'")


if __name__ == "__main__":
    main()
