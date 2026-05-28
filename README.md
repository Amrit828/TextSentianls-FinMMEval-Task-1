# TextSentianls-FinMMEval-Task-1

Code for our FinMMEval 2026 Task 1 submission: a multilingual routed retrieval-augmented system for financial multiple-choice question answering.

## Included

- Five evaluated Task 1 pipeline variants
- Task 1 submission runner scripts

## Core variants

- `task1_faiss_rag_openai.py`
- `task1_faiss_rag_openai_v2.py`
- `task1_groq_routed_ensemble_v3.py`
- `task1_groq_routed_ensemble_v4.py`
- `task1_groq_confidence_v5.py`

## Requirements

Set the needed API keys as environment variables before running:

- `OPENAI_API_KEY`
- `GROQ_API_KEY`

The scripts do not contain hardcoded API keys.
