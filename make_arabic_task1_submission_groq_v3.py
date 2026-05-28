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

import task1_faiss_rag_openai as base
import task1_groq_routed_ensemble_v3 as v3


# Update these paths if your Arabic files use different names.
QUESTIONS_PATH = Path(r"C:\Users\amrit\Documents\ResearchLab\CLEF\arabic_task1_final_public.jsonl")
TEMPLATE_PATH = Path(r"C:\Users\amrit\Documents\ResearchLab\CLEF\arabic_task1_final_submission_template.json")
OUTPUT_PATH = Path(r"C:\Users\amrit\Documents\ResearchLab\CLEF\arabic_task1_final_submission_groq_v3.json")


def load_questions():
    if not QUESTIONS_PATH.exists():
        raise RuntimeError(f"Arabic public questions file not found: {QUESTIONS_PATH}")
    rows = []
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_template():
    if not TEMPLATE_PATH.exists():
        raise RuntimeError(f"Arabic submission template not found: {TEMPLATE_PATH}")
    with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_query_example(row: dict) -> base.MCQExample:
    return base.MCQExample(
        row_id=row["id"],
        dataset_id="arabic_task1_final_public",
        split="public",
        language="Arabic",
        question=row["question"],
        options=row["options"],
        answer=None,
        reason=None,
        query=None,
        raw=row,
    )


def main():
    client = v3.get_groq_client()
    base.INDEX_DIR = v3.INDEX_DIR
    index, corpus_examples = base.load_index_and_metadata()
    embedder = base.get_embedder()

    questions = load_questions()
    template = load_template()
    template_ids = {row["id"] for row in template}
    predictions = []

    print(f"Loaded {len(questions)} Arabic public/dev questions")

    for i, row in enumerate(questions, start=1):
        example = build_query_example(row)
        route_info = v3.route_question(client, example)
        retrieved = v3.retrieve_for_route(
            example=example,
            route_info=route_info,
            embedder=embedder,
            index=index,
            corpus_examples=corpus_examples,
            excluded_target_ids=set(),
        )
        qwen_result = v3.score_with_model(client, v3.FACTUAL_MODEL, route_info, example, retrieved)
        gptoss_result = v3.score_with_model(client, v3.REASONING_MODEL, route_info, example, retrieved)
        prediction, _merged_scores = v3.ensemble_scores(route_info, example, qwen_result, gptoss_result)

        if prediction not in example.options:
            prediction = list(example.options.keys())[0]

        if row["id"] not in template_ids:
            raise RuntimeError(f"Question id {row['id']} not found in template")

        predictions.append({
            "id": row["id"],
            "prediction": prediction,
        })

        if i == 1 or i % 10 == 0 or i == len(questions):
            print(
                f"[{i}/{len(questions)}] id={row['id']} route={route_info['route']} "
                f"pred={prediction} retrieved={len(retrieved)}"
            )

    if len(predictions) != len(template):
        raise RuntimeError(f"Filled {len(predictions)} rows but template has {len(template)} rows")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Saved submission file to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
