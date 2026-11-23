"""Inference-only ablations on one trained run (SPEC §6: E4, E5, E5b).

Each stage varies one matching setting; its winner on VALIDATION F1 (ties: validation
full-retrieval MRR) is carried into the next stage:

    E4   class representation   max | mean                      (cosine, argmax)
    E5   scoring x selection    {cosine, csls} x {argmax, mutual, stable_marriage}
    E5b  neighbor re-scoring    λ in {0.1, 0.2, 0.3}             (λ = 0 is the E5 winner)

Every evaluation appends a row to results/summary.csv (experiment = E4 / E5 / E5b).
The chosen settings are written to results/inference_ablations.json.

Usage:  uv run python scripts/inference_ablations.py runs/<stamp>-final
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contrastive_om.config import load_config
from contrastive_om.pipeline import evaluate_run, load_task_ontologies
from contrastive_om.train import load_encoder

log = logging.getLogger("inference_ablations")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--lambdas", nargs="*", type=float, default=[0.1, 0.2, 0.3])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_dir = Path(args.run_dir)
    base_cfg = load_config(run_dir / "config.yaml")
    onts = load_task_ontologies(base_cfg)
    model = load_encoder(base_cfg, path=str(run_dir / "model"))
    results: list[dict] = []

    def evaluate(stage: str, match: dict) -> dict:
        cfg = load_config(run_dir / "config.yaml", overrides={"experiment": stage, "match": match})
        m = evaluate_run(cfg, run_dir, model=model, ontologies=onts)
        row = {"stage": stage, **match, "val_f1": m["val_f1"], "val_full_mrr": m["val_full_mrr"],
               "test_f1": m["test_f1"], "theta": m["theta"]}
        results.append(row)
        log.info("%s %s -> val F1 %.4f", stage, match, m["val_f1"])
        return row

    def best(rows: list[dict]) -> dict:
        return max(rows, key=lambda r: (r["val_f1"], r["val_full_mrr"]))

    fixed = {"scoring": "cosine", "selection": "argmax", "neighbor_lambda": 0.0}
    e4 = [evaluate("E4", {**fixed, "class_repr": repr_}) for repr_ in ("max", "mean")]
    class_repr = best(e4)["class_repr"]

    e5 = [r for r in e4 if r["class_repr"] == class_repr]  # (cosine, argmax) already evaluated
    for scoring in ("cosine", "csls"):
        for selection in ("argmax", "mutual", "stable_marriage"):
            if (scoring, selection) != ("cosine", "argmax"):
                e5.append(evaluate("E5", {"class_repr": class_repr, "scoring": scoring,
                                          "selection": selection, "neighbor_lambda": 0.0}))
    w5 = best(e5)

    e5b = [w5] + [evaluate("E5b", {"class_repr": class_repr, "scoring": w5["scoring"],
                                   "selection": w5["selection"], "neighbor_lambda": lam}) for lam in args.lambdas]
    w5b = best(e5b)

    chosen = {k: w5b[k] for k in ("class_repr", "scoring", "selection", "neighbor_lambda")}
    out = {"run": str(run_dir), "chosen_on_validation": chosen, "evaluations": results}
    dest = Path(base_cfg.output.results_dir) / "inference_ablations.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"\n{'stage':5} {'repr':5} {'scoring':7} {'selection':16} {'λ':>4} {'θ':>6} {'val F1':>7} {'val MRR':>7} {'test F1':>7}")
    for r in results:
        print(f"{r['stage']:5} {r['class_repr']:5} {r['scoring']:7} {r['selection']:16} {r['neighbor_lambda']:4.1f} "
              f"{r['theta']:6.3f} {r['val_f1']:7.4f} {r['val_full_mrr']:7.4f} {r['test_f1']:7.4f}")
    print(f"\nchosen on validation: {chosen}\nwritten to {dest}")


if __name__ == "__main__":
    main()
