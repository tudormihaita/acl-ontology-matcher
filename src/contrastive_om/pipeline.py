"""Run-level orchestration: load the task, mine, train, match and evaluate (SPEC §5).

``evaluate_run`` scores one configuration on the task:
1. class scores -> selection -> candidate mappings (before θ);
2. θ chosen on **validation** (test as null references), test P/R/F1 at that θ;
3. local ranking (official test candidates), full retrieval (gold among all targets);
4. embedding-space alignment / uniformity; timings; ``metrics.json`` + one ``summary.csv`` row.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from contrastive_om import evaluate as ev
from contrastive_om.match import jaccard_class_scores, match, string_match
from contrastive_om.ontology import Ontology, load_from_config, restrict
from contrastive_om.represent import ClassEmbeddings, embed_ontology

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- task loading


def load_task_ontologies(cfg) -> dict[str, Ontology]:
    onts = {which: load_from_config(cfg, which) for which in ("source", "target")}
    if cfg.task.get("subsample"):
        onts = subsample_task(cfg, onts)
    return onts


def subsample_task(cfg, onts: dict) -> dict:
    """Smoke configuration only: keep ``reference_pairs`` mapped pairs (so that matching has
    something to find) plus random classes, up to ``classes`` per ontology. Never reported."""
    sub = cfg.task.subsample
    rng = random.Random(int(sub.get("seed", 0)))
    refs = [(s, t) for s, t in ev.read_mappings(cfg.task.full_refs)
            if s in onts["source"].classes and t in onts["target"].classes]
    chosen = rng.sample(sorted(refs), min(int(sub.reference_pairs), len(refs)))
    keep = {"source": {s for s, _ in chosen}, "target": {t for _, t in chosen}}
    for which, ont in onts.items():
        others = sorted(set(ont.classes) - keep[which])
        need = max(0, int(sub.classes) - len(keep[which]))
        keep[which] |= set(rng.sample(others, min(need, len(others))))
        onts[which] = restrict(ont, keep[which])
    log.info("smoke subsample: %d source / %d target classes", len(onts["source"]), len(onts["target"]))
    return onts


def mine_and_save(cfg, ontologies=None, force: bool = False):
    """Mine pairs (cached by config hash) and return (rows, stats, path)."""
    from contrastive_om.pairs import load_rows, mine_pairs, pairs_path, save_rows

    path = pairs_path(cfg)
    if path.exists() and not force:
        log.info("using cached pairs %s", path)
        return load_rows(path), json.loads(path.with_suffix(".stats.json").read_text()), path
    ontologies = ontologies or load_task_ontologies(cfg)
    refs = set(ev.read_mappings(cfg.task.full_refs))  # only for the leakage assertion
    fixes = {w: cfg.task.get(f"{w}_label_fixes") for w in ontologies}
    rows, stats, lexicons = mine_pairs(cfg, ontologies, references=refs, label_fixes=fixes)
    save_rows(rows, stats, lexicons, path)
    return rows, stats, path


def train_run(cfg, rows, run_dir: Path):
    """Train, then verify that the saved checkpoint reloads to identical embeddings.
    Returns ``(summary, model)``."""
    from contrastive_om.train import load_encoder, loss_summary, read_train_log, train

    timing, model = train(cfg, rows, run_dir)
    summary = loss_summary(read_train_log(run_dir))
    probe = [rows[0].anchor, rows[0].positive, "alzheimer disease"]
    trained = model.encode(probe, normalize_embeddings=True)
    reloaded = load_encoder(cfg, path=str(run_dir / "model")).encode(probe, normalize_embeddings=True)
    summary.update({
        "checkpoint_reloads": bool(np.allclose(trained, reloaded, atol=1e-5)),
        "probe_positive_cosine": round(float(reloaded[0] @ reloaded[1]), 4),
    })
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return {**timing, **summary}, model


def run(cfg) -> tuple[Path, dict]:
    """mine -> train -> evaluate (``run`` command). Epoch 0 / baseline skip mining and training."""
    from contrastive_om.train import make_run_dir

    run_dir = make_run_dir(cfg)
    onts = load_task_ontologies(cfg)
    model = None
    if cfg.match.method == "embedding" and cfg.train.enabled:
        rows, _, _ = mine_and_save(cfg, onts)
        summary, model = train_run(cfg, rows, run_dir)
        log.info("training: %s", summary)
    return run_dir, evaluate_run(cfg, run_dir, model=model, ontologies=onts)


# --------------------------------------------------------------------------- helpers


def _parent_indices(ont: Ontology, emb: ClassEmbeddings) -> list[list[int]]:
    index = emb.index
    return [sorted(index[p] for p in ont.informative_parents(iri) if p in index) for iri in emb.iris]


def _alignable_mask(ont: Ontology, emb: ClassEmbeddings) -> np.ndarray:
    return np.array([ont[iri].alignable for iri in emb.iris], dtype=bool)


def _full_ranks(scores: np.ndarray, pairs, src_index, tgt_index) -> list[int]:
    """Rank of the gold target among all targets of its source row (pessimistic ties)."""
    ranks = []
    for s, t in pairs:
        row = scores[src_index[s]]
        gold = row[tgt_index[t]]
        ranks.append(int((row >= gold).sum()) if np.isfinite(gold) else len(row))
    return ranks


def _local_ranks(score_row_fn, cands, tgt_index) -> list[int]:
    ranks = []
    for c in cands:
        row = score_row_fn(c.src)
        scores = [float(row[tgt_index[t]]) if t in tgt_index else float("nan") for t in c.candidates]
        ranks.append(ev.rank_from_scores(c.candidates, scores, c.gold))
    return ranks


def _summary_row(cfg, run_dir: Path, metrics: dict) -> dict:
    keys = {
        "experiment": cfg.get("experiment", ""), "name": cfg.name, "run": str(run_dir),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "method": cfg.match.method, "encoder": cfg.model.encoder,
        "trained": bool(cfg.train.enabled) and cfg.match.method == "embedding",
        "positives": "+".join(cfg.pairs.positives), "hard_negatives": "+".join(cfg.pairs.hard_negatives or []) or "none",
        "loss": cfg.train.loss, "temperature": cfg.train.temperature, "batch_size": cfg.train.batch_size,
        "seed": cfg.seed, "class_repr": cfg.match.class_repr, "scoring": cfg.match.scoring,
        "selection": cfg.match.selection, "neighbor_lambda": cfg.match.get("neighbor_lambda", 0.0),
    }
    return {**keys, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}}


# --------------------------------------------------------------------------- evaluation


def evaluate_run(cfg, run_dir: Path, model=None, ontologies: dict | None = None, append_summary: bool = True) -> dict:
    run_dir = Path(run_dir)
    eval_dir = run_dir / f"eval-{cfg.hash('match')}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    onts = ontologies or load_task_ontologies(cfg)
    src_ont, tgt_ont = onts["source"], onts["target"]

    def present(pairs):  # all pairs on the full task; the smoke subsample drops some
        return [(s, t) for s, t in pairs if s in src_ont.classes and t in tgt_ont.classes]

    val, test = present(ev.read_mappings(cfg.task.val_refs)), present(ev.read_mappings(cfg.task.test_refs))
    cands = [c for c in ev.read_candidates(cfg.task.test_cands)
             if c.src in src_ont.classes and c.gold in tgt_ont.classes]
    metrics: dict = {"n_val_refs": len(val), "n_test_refs": len(test), "n_test_cands": len(cands)}
    timing: dict = {}

    # ---- scores and candidate mappings
    if cfg.match.method == "string_match":
        t0 = time.perf_counter()
        src_lab = {c.iri: c.labels for c in src_ont if c.alignable and c.labels}
        tgt_lab = {c.iri: c.labels for c in tgt_ont if c.alignable and c.labels}
        mappings = string_match(src_lab, tgt_lab)
        src_emb = _label_only(src_ont)
        tgt_emb = _label_only(tgt_ont)
        src_index, tgt_index = src_emb.index, tgt_emb.index
        rows_needed = sorted({src_index[s] for s, _ in val + test} | {src_index[c.src] for c in cands})
        jac = jaccard_class_scores(src_emb, tgt_emb, rows_needed)
        tgt_ok = _alignable_mask(tgt_ont, tgt_emb)
        jac[:, ~tgt_ok] = -np.inf
        row_of = {r: i for i, r in enumerate(rows_needed)}
        scores_for = lambda s: jac[row_of[src_index[s]]]  # noqa: E731
        timing["match_s"] = round(time.perf_counter() - t0, 2)
        full_scores = None
    else:
        from contrastive_om.train import load_encoder

        model = model or load_encoder(cfg, path=_model_path(cfg, run_dir))
        t0 = time.perf_counter()
        src_emb = embed_ontology(src_ont, model, run_dir / "cache", tag=f"src-{src_ont.name}")
        tgt_emb = embed_ontology(tgt_ont, model, run_dir / "cache", tag=f"tgt-{tgt_ont.name}")
        timing["encode_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        result = match(src_emb, tgt_emb, _alignable_mask(src_ont, src_emb), _alignable_mask(tgt_ont, tgt_emb),
                       cfg.match, _parent_indices(src_ont, src_emb), _parent_indices(tgt_ont, tgt_emb))
        timing["match_s"] = round(time.perf_counter() - t0, 2)
        mappings = result.mappings
        src_index, tgt_index = src_emb.index, tgt_emb.index
        full_scores = result.scores
        scores_for = lambda s: full_scores[src_index[s]]  # noqa: E731

    # ---- global matching: θ on validation, report on test
    if cfg.match.method == "string_match":
        grid = [1.0]
    elif cfg.match.scoring == "csls":
        grid = ev.percentile_grid(list(mappings.values()))
    else:
        grid = ev.threshold_grid(*cfg.match.threshold_grid)
    curve = ev.threshold_curve(mappings, val, test, grid)
    theta, best = ev.select_threshold(curve)
    kept = ev.apply_threshold(mappings, theta)
    test_m = ev.global_matching(kept, test, null_references=val)
    val_m = ev.global_matching(kept, val, null_references=test)
    metrics.update({"theta": theta, "n_candidate_mappings": len(mappings), "n_mappings": len(kept)})
    metrics.update(val_m.as_dict("val_"))
    metrics.update(test_m.as_dict("test_"))
    ci = ev.bootstrap_prf(kept, test, val, n_resamples=1000, seed=int(cfg.seed))
    metrics.update({"test_f1_ci_low": ci["f1_ci_low"], "test_f1_ci_high": ci["f1_ci_high"]})

    # ---- ranking
    local = _local_ranks(scores_for, cands, tgt_index)
    metrics.update(ev.ranking_metrics(local, ks=(1, 5), prefix="local_"))
    mrr_ci = ev.bootstrap_mean([1 / r for r in local], seed=int(cfg.seed))
    metrics.update({"local_mrr_ci_low": mrr_ci["ci_low"], "local_mrr_ci_high": mrr_ci["ci_high"]})
    if full_scores is not None:
        metrics.update(ev.ranking_metrics(_full_ranks(full_scores, test, src_index, tgt_index), (1, 5, 10), "full_"))
        metrics.update(ev.ranking_metrics(_full_ranks(full_scores, val, src_index, tgt_index), (1, 5, 10), "val_full_"))
    else:
        full = [_rank_in_row(scores_for(s), tgt_index[t]) for s, t in test]
        metrics.update(ev.ranking_metrics(full, (1, 5, 10), "full_"))
        metrics.update(ev.ranking_metrics([_rank_in_row(scores_for(s), tgt_index[t]) for s, t in val],
                                          (1, 5, 10), "val_full_"))

    # ---- embedding space
    if cfg.match.method != "string_match" and test:
        si = [src_index[s] for s, _ in test]
        ti = [tgt_index[t] for _, t in test]
        metrics["alignment"] = ev.alignment(src_emb.class_vecs[si], tgt_emb.class_vecs[ti])
        rng = np.random.default_rng(int(cfg.seed))
        pool = np.concatenate([src_emb.label_vecs, tgt_emb.label_vecs])
        sample = pool[rng.choice(len(pool), size=min(5000, len(pool)), replace=False)]
        metrics["uniformity"] = ev.uniformity(sample)

    # ---- timings, files, summary
    train_timing = run_dir / "timing.json"
    if train_timing.exists():
        timing["train_s"] = json.loads(train_timing.read_text()).get("train_s")
    metrics.update(timing)
    (eval_dir / "threshold_curve.json").write_text(json.dumps(curve, indent=1))
    ev.write_scored_mappings(kept, eval_dir / "mappings.tsv")
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (eval_dir / "config.yaml").write_text(__import__("yaml").safe_dump(cfg.to_dict(), sort_keys=False))
    if append_summary and cfg.output.get("append_summary", True):
        ev.append_summary(_summary_row(cfg, run_dir, metrics), Path(cfg.output.results_dir) / "summary.csv")
    log.info("θ=%.3f test P/R/F1 %.3f/%.3f/%.3f | local MRR %.3f", theta, test_m.precision, test_m.recall,
             test_m.f1, metrics["local_mrr"])
    return metrics


def _rank_in_row(row: np.ndarray, gold_idx: int) -> int:
    gold = row[gold_idx]
    return int((row >= gold).sum()) if np.isfinite(gold) else len(row)


def _label_only(ont: Ontology) -> ClassEmbeddings:
    """Label layout without vectors (for the string-match baseline)."""
    iris = [c.iri for c in ont if c.labels]
    labels, offsets = [], []
    for iri in iris:
        offsets.append(len(labels))
        labels.extend(ont[iri].labels)
    empty = np.zeros((0, 0), np.float32)
    return ClassEmbeddings(iris, labels, empty, np.asarray(offsets, np.int64), empty)


def _model_path(cfg, run_dir: Path) -> str | None:
    """The run's trained model if present; otherwise ``None`` (= the pre-trained encoder, epoch 0)."""
    path = run_dir / "model"
    if path.exists():
        return str(path)
    if cfg.train.enabled and cfg.match.method == "embedding":
        raise FileNotFoundError(f"{path} not found: train first, or set train.enabled=false for epoch 0")
    return None
