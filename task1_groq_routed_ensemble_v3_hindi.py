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
from pathlib import Path

import faiss

import task1_faiss_rag_openai as base
import task1_groq_routed_ensemble_v3 as v3


HINDI_INDEX_DIR = "task1_faiss_store_groq_v3_hindi"
SOURCE_V3_INDEX_DIR = "task1_faiss_store_groq_v3"
HINDI_SOURCE = {
    "dataset_id": "bharatgenai/BhashaBench-Finance",
    "config": "Hindi",
    "split": "test",
}

ROUTER_MODEL = v3.ROUTER_MODEL
FACTUAL_MODEL = v3.FACTUAL_MODEL
REASONING_MODEL = v3.REASONING_MODEL

RUN_MODE = "eval"
FORCE_REBUILD_INDEX = False
LIMIT = 50
PREDICTIONS_PATH = "task1_groq_routed_ensemble_v3_hindi_predictions.jsonl"


def get_groq_client():
    return v3.get_groq_client()


def route_question(client, example):
    return v3.route_question(client, example)


def score_with_model(client, model, route_info, example, retrieved):
    return v3.score_with_model(client, model, route_info, example, retrieved)


def ensemble_scores(route_info, example, qwen_result, gptoss_result):
    return v3.ensemble_scores(route_info, example, qwen_result, gptoss_result)


def retrieve_for_route(example, route_info, embedder, index, corpus_examples, excluded_target_ids):
    return v3.retrieve_for_route(example, route_info, embedder, index, corpus_examples, excluded_target_ids)


def _load_existing_v3_metadata() -> list[base.MCQExample]:
    metadata_path = Path(SOURCE_V3_INDEX_DIR) / "task1_metadata.jsonl"
    if not metadata_path.exists():
        raise RuntimeError(
            f"Base v3 metadata not found at {metadata_path}. Build the original v3 store first."
        )
    return base.load_metadata(metadata_path)


def _load_hindi_examples() -> list[base.MCQExample]:
    print(
        "Loading Hindi corpus from "
        f"{HINDI_SOURCE['dataset_id']}:{HINDI_SOURCE['config']}:{HINDI_SOURCE['split']}"
    )
    return base.load_mcq_dataset(
        HINDI_SOURCE["dataset_id"],
        HINDI_SOURCE["config"],
        HINDI_SOURCE["split"],
        language_filters=["Hindi"],
        max_rows=None,
    )


def _dedupe_examples(examples: list[base.MCQExample]) -> list[base.MCQExample]:
    seen = set()
    deduped = []
    for example in examples:
        key = (example.dataset_id, example.split, example.row_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def build_and_save_hindi_augmented_index() -> tuple[faiss.Index, list[base.MCQExample]]:
    index_dir = Path(HINDI_INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)

    base_examples = _load_existing_v3_metadata()
    hindi_examples = _load_hindi_examples()
    corpus_examples = _dedupe_examples(base_examples + hindi_examples)
    print(
        f"Building Hindi-augmented FAISS store with {len(base_examples)} base examples "
        f"+ {len(hindi_examples)} Hindi examples = {len(corpus_examples)} total"
    )

    texts = [base.example_to_passage(x) for x in corpus_examples]
    embedder = base.get_embedder()
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "task1.index"))
    base.save_metadata(index_dir / "task1_metadata.jsonl", corpus_examples)
    manifest = {
        "embed_model": base.EMBED_MODEL,
        "bootstrapped_from": SOURCE_V3_INDEX_DIR,
        "added_source": HINDI_SOURCE,
        "index_size": len(corpus_examples),
        "dimension": int(embeddings.shape[1]),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved Hindi-augmented FAISS index and metadata to {index_dir.resolve()}")
    return index, corpus_examples


def load_index_and_metadata() -> tuple[faiss.Index, list[base.MCQExample]]:
    index_dir = Path(HINDI_INDEX_DIR)
    index_path = index_dir / "task1.index"
    metadata_path = index_dir / "task1_metadata.jsonl"
    if FORCE_REBUILD_INDEX or not index_path.exists() or not metadata_path.exists():
        return build_and_save_hindi_augmented_index()
    index = faiss.read_index(str(index_path))
    metadata = base.load_metadata(metadata_path)
    print(f"Loaded Hindi-augmented FAISS index with {index.ntotal} vectors from {index_dir.resolve()}")
    return index, metadata


def run_eval_local():
    client = get_groq_client()
    base.INDEX_DIR = HINDI_INDEX_DIR
    index, corpus_examples = load_index_and_metadata()
    target_examples = base.load_target_examples()
    target_examples = base.sample_target_examples(target_examples)
    excluded_target_ids = base.build_excluded_target_ids(target_examples)
    embedder = base.get_embedder()

    route_counts = {"factual_rag": 0, "reasoning_math": 0, "exhibit_best_effort": 0}
    retrieval_hits = 0
    predictions = []

    print(f"Testing on {len(target_examples)} question(s)")
    for i, example in enumerate(target_examples, start=1):
        route_info = route_question(client, example)
        route_counts[route_info["route"]] += 1
        retrieved = retrieve_for_route(example, route_info, embedder, index, corpus_examples, excluded_target_ids)
        if retrieved:
            retrieval_hits += 1
        qwen_result = score_with_model(client, FACTUAL_MODEL, route_info, example, retrieved)
        gptoss_result = score_with_model(client, REASONING_MODEL, route_info, example, retrieved)
        prediction, merged_scores = ensemble_scores(route_info, example, qwen_result, gptoss_result)
        if prediction not in example.options:
            prediction = list(example.options.keys())[0]
        predictions.append(
            {
                "id": example.row_id,
                "dataset_id": example.dataset_id,
                "split": example.split,
                "language": example.language,
                "prediction": prediction,
                "question": example.question,
                "retrieved_ids": [x.row_id for x, _ in retrieved],
                "retrieved_datasets": [x.dataset_id for x, _ in retrieved],
                "gold": example.answer,
                "correct": prediction == example.answer if example.answer else None,
                "route": route_info["route"],
                "difficulty": route_info["difficulty"],
                "scores": merged_scores,
            }
        )
        if i == 1 or i % 10 == 0 or i == len(target_examples):
            print(
                f"[{i}/{len(target_examples)}] route={route_info['route']} pred={prediction} "
                f"retrieved={len(retrieved)} gold={example.answer}"
            )

    with Path(PREDICTIONS_PATH).open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    valid = sum(1 for row in predictions if row["prediction"] in {"A", "B", "C", "D", "E"})
    scored = [row for row in predictions if row.get("correct") is not None]
    correct = sum(1 for row in scored if row["correct"])
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print(f"Retrieval hits: {retrieval_hits}/{len(predictions)} = {retrieval_hits / max(len(predictions), 1):.4f}")
    print(f"Route counts: {route_counts}")
    print(f"Valid answer rate: {valid}/{len(predictions)} = {valid / max(len(predictions), 1):.4f}")
    if scored:
        print(f"Accuracy: {correct}/{len(scored)} = {correct / len(scored):.4f}")


def main():
    if RUN_MODE == "build":
        build_and_save_hindi_augmented_index()
    elif RUN_MODE == "eval":
        run_eval_local()
    else:
        raise RuntimeError(f"Unsupported RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
