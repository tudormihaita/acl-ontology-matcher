# CLAUDE.md

This project is a contrastive bi-encoder for unsupervised ontology matching (ACL course, RR2). The full spec is in `SPEC.md`. Read it at the start of every session.

## How to work
- Implement **one phase at a time** (SPEC §9). Before you start, state which phase you are on. At the end, run that phase's acceptance checks.
- Do not move to the next phase until the checks pass. Commit after each phase as `phase N: <summary>`.
- Always run `configs/smoke.yaml` end to end before launching full-task runs.
- Ask before changing scope, the method, the metrics or the protocol. Extensions (SPEC §10) start only after Phase 7, and only when the user asks.
- If the data download fails or the dataset layout differs from the spec, stop and report what you found. Do not guess.

## Non-negotiables
- Reference mappings are **never** used for training.
- **Test** references are never used for any decision: thresholds, hyperparameters or early stopping. Thresholds are chosen on the validation split only.
- Batches are **class-unique** and **per-ontology** (SPEC §4.2). These rules are enforced by tests.
- The core pipeline has no rule-based extension and no logical repair.

## Environment
- Mac mini M4: `device: auto` resolves to `mps`. Export `PYTORCH_ENABLE_MPS_FALLBACK=1`. Train in fp32 (no mixed precision on MPS).
- Python 3.11 venv in `.venv`. Install with `pip install -e ".[dev]"`.
- Tests: `pytest -q`. Every phase keeps the suite green.

## Commands
- `python -m contrastive_om.cli run --config configs/<name>.yaml`: mine, train, match and evaluate.
- `python -m contrastive_om.cli match --config C --model PATH --source X.owl --target Y.owl`: the demo.
- Results append to `results/summary.csv`. Each run writes to `runs/<timestamp>-<name>/`.
