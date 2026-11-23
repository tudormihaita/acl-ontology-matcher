# CLAUDE.md

This project is a contrastive bi-encoder for unsupervised ontology matching (ACL course, RR2). The full spec is in `docs/SPEC.md`. Read it at the start of every session.

## How to work
- Implement **one phase at a time** (SPEC §9). Before you start, state which phase you are on. At the end, run that phase's acceptance checks.
- Do not move to the next phase until the checks pass. The user makes all git commits; do not commit.
- Always run `configs/smoke.yaml` end to end before launching full-task runs.
- Ask before changing scope, the method, the metrics or the protocol. Extensions (SPEC §10) start only after Phase 8 (presentation), and only when the user asks.
- If the data download fails or the dataset layout differs from the spec, stop and report what you found. Do not guess.

## Non-negotiables
- Reference mappings are **never** used for training.
- **Test** references are never used for any decision: thresholds, hyperparameters, early stopping or picking the best setting of an experiment. These choices use the validation split only.
- No leakage sources: cross-references (`hasDbXref`, `exactMatch`, ...), UMLS/MONDO/DOID/MeSH/Wikidata, SapBERT, or LLM-generated synonyms (SPEC §3.3).
- Batches are **class-unique**, **per-ontology**, and contain **no hard negative from an in-batch anchor class** and **no text conflicts** (SPEC §4.2.3). These rules are enforced by tests.
- The core pipeline has no rule-based extension and no logical repair.

## Environment
- Mac mini M4: `device: auto` resolves to `mps`. Export `PYTORCH_ENABLE_MPS_FALLBACK=1`. Train in fp32 (no mixed precision on MPS).
- Python 3.12, managed with uv (`.venv`). Install with `uv sync --extra dev`; run with `uv run ...`.
- Tests: `uv run pytest -q`. Every phase keeps the suite green.

## Commands
- `uv run python -m contrastive_om.cli run --config configs/<name>.yaml`: mine, train, match and evaluate.
- `uv run python -m contrastive_om.cli match --config C --model PATH --source X.owl --target Y.owl`: the demo.
- Results append to `results/summary.csv`. Each run writes to `runs/<timestamp>-<name>/`.
