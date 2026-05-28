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

print("before v3")

import task1_faiss_rag_openai as base
import task1_groq_routed_ensemble_v3 as v3

print("inside script")

# -----------------------------
# Change only these values.
# -----------------------------
LANGUAGE = "Arabic"  # "English", "Chinese", "Arabic"

LANGUAGE_CONFIGS = {
    "English": {
        "questions_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\english_task1_final_test_public.jsonl",
        "template_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\english_task1_final_test_submission_template.json",
        "output_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\english_task1_final_test_submission_groq_v3_combined.json",
        "debug_output_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\english_task1_final_test_debug_groq_v3_combined.jsonl",
    },
    "Chinese": {
        "questions_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\chinese_task1_final_test_public.jsonl",
        "template_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\chinese_task1_final_test_submission_template.json",
        "output_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\chinese_task1_final_test_submission_groq_v3_combined.json",
        "debug_output_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\chinese_task1_final_test_debug_groq_v3_combined.jsonl",
    },
    "Arabic": {
        "questions_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\arabic_task1_final_test_public.jsonl",
        "template_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\arabic_task1_final_test_submission_template.json",
        "output_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\arabic_task1_final_test_submission_groq_v3_combined.json",
        "debug_output_path": r"C:\Users\amrit\Documents\ResearchLab\CLEF\arabic_task1_final_test_debug_groq_v3_combined.jsonl",
    },
}


def get_paths():
    if LANGUAGE not in LANGUAGE_CONFIGS:
        raise RuntimeError(f"Unsupported LANGUAGE: {LANGUAGE}")
    cfg = LANGUAGE_CONFIGS[LANGUAGE]
    return (
        Path(cfg["questions_path"]),
        Path(cfg["template_path"]),
        Path(cfg["output_path"]),
        Path(cfg["debug_output_path"]),
    )


def load_questions(path: Path):
    if not path.exists():
        raise RuntimeError(f"Questions file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_template(path: Path):
    if not path.exists():
        raise RuntimeError(f"Template file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_debug_predictions(debug_output_path: Path) -> dict[str, dict]:
    existing = {}
    if not debug_output_path.exists():
        return existing
    with debug_output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            existing[row["id"]] = row
    return existing


def write_submission_from_debug(template: list[dict], debug_predictions: dict[str, dict], output_path: Path) -> None:
    ordered = []
    for row in template:
        qid = row["id"]
        if qid not in debug_predictions:
            continue
        ordered.append({
            "id": qid,
            "prediction": debug_predictions[qid]["prediction"],
        })
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, separators=(",", ":"))


def build_query_example(row: dict) -> base.MCQExample:
    return base.MCQExample(
        row_id=row["id"],
        dataset_id=f"{LANGUAGE.lower()}_task1_dev_public",
        split="public",
        language=LANGUAGE,
        question=row["question"],
        options=row["options"],
        answer=None,
        reason=None,
        query=None,
        raw=row,
    )


def main():
    questions_path, template_path, output_path, debug_output_path = get_paths()

    client = v3.get_groq_client()
    base.INDEX_DIR = v3.INDEX_DIR
    index, corpus_examples = base.load_index_and_metadata()
    embedder = base.get_embedder()

    questions = load_questions(questions_path)
    template = load_template(template_path)
    template_ids = {row["id"] for row in template}
    existing_debug = load_existing_debug_predictions(debug_output_path)
    completed_ids = set(existing_debug.keys())
    retrieval_hits = 0

    print(f"Loaded {len(questions)} {LANGUAGE} dev/public questions")
    print(f"Found {len(completed_ids)} existing completed predictions in {debug_output_path}")

    with debug_output_path.open("a", encoding="utf-8") as debug_f:
        for i, row in enumerate(questions, start=1):
            if row["id"] in completed_ids:
                existing_row = existing_debug[row["id"]]
                if existing_row.get("retrieved_ids"):
                    retrieval_hits += 1
                if i == 1 or i % 10 == 0 or i == len(questions):
                    print(f"[{i}/{len(questions)}] id={row['id']} already completed, skipping")
                continue

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
            if retrieved:
                retrieval_hits += 1

            qwen_result = v3.score_with_model(client, v3.FACTUAL_MODEL, route_info, example, retrieved)
            gptoss_result = v3.score_with_model(client, v3.REASONING_MODEL, route_info, example, retrieved)
            prediction, merged_scores = v3.ensemble_scores(route_info, example, qwen_result, gptoss_result)

            if prediction not in example.options:
                prediction = list(example.options.keys())[0]

            if row["id"] not in template_ids:
                raise RuntimeError(f"Question id {row['id']} not found in template")

            debug_row = {
                "id": row["id"],
                "language": LANGUAGE,
                "route": route_info["route"],
                "difficulty": route_info["difficulty"],
                "prediction": prediction,
                "scores": merged_scores,
                "retrieved_ids": [item.row_id for item, _ in retrieved],
                "retrieved_datasets": [item.dataset_id for item, _ in retrieved],
                "model_votes": {
                    "qwen": qwen_result["best_answer"],
                    "gpt_oss": gptoss_result["best_answer"],
                },
            }
            debug_f.write(json.dumps(debug_row, ensure_ascii=False) + "\n")
            debug_f.flush()
            existing_debug[row["id"]] = debug_row
            completed_ids.add(row["id"])

            if i == 1 or i % 10 == 0 or i == len(questions):
                print(
                    f"[{i}/{len(questions)}] id={row['id']} route={route_info['route']} "
                    f"pred={prediction} retrieved={len(retrieved)}"
                )
            write_submission_from_debug(template, existing_debug, output_path)

    if len(existing_debug) != len(template):
        missing = len(template) - len(existing_debug)
        print(f"Submission is partial: {missing} item(s) still missing")
    write_submission_from_debug(template, existing_debug, output_path)

    print(f"\nSaved submission file to {output_path}")
    print(f"Saved debug file to {debug_output_path}")
    print(f"Retrieval hits: {retrieval_hits}/{len(questions)} = {retrieval_hits / max(len(questions), 1):.4f}")


if __name__ == "__main__":
    main()
