"""Evaluation: global matching, ranking, threshold selection, embedding-space metrics, bootstrap.

Conventions (SPEC §5):
- **Null references.** When scoring on one split, predictions that are *exactly* a mapping of
  the other split are removed before computing P/R/F1 (DeepOnto / Bio-ML convention). The
  reference set itself is the evaluated split only.
- **Ranking ties are pessimistic.** The gold target is ranked after every candidate with an
  equal or higher score, so methods with many tied scores get no free credit.
- **Thresholds are chosen on validation only.** :func:`select_threshold` never sees test refs.
"""

from __future__ import annotations

import ast
import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)

Pair = tuple[str, str]


# --------------------------------------------------------------------------- I/O


def read_mappings(path: str | Path) -> list[Pair]:
    """Read a ``SrcEntity, TgtEntity[, Score]`` TSV into a list of (src, tgt) pairs."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [(row["SrcEntity"], row["TgtEntity"]) for row in reader]


def read_scored_mappings(path: str | Path) -> dict[Pair, float]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return {(row["SrcEntity"], row["TgtEntity"]): float(row["Score"]) for row in reader}


def write_scored_mappings(mappings: dict[Pair, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["SrcEntity", "TgtEntity", "Score"])
        for (s, t), score in sorted(mappings.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow([s, t, f"{score:.6f}"])


@dataclass(frozen=True)
class CandidateRow:
    src: str
    gold: str
    candidates: tuple[str, ...]  # includes the gold target


def read_candidates(path: str | Path) -> list[CandidateRow]:
    """Read Bio-ML ``test.cands.tsv`` (``TgtCandidates`` is a Python tuple literal)."""
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            cands = tuple(ast.literal_eval(row["TgtCandidates"]))
            if row["TgtEntity"] not in cands:
                raise ValueError(f"gold target missing from candidates for {row['SrcEntity']}")
            rows.append(CandidateRow(row["SrcEntity"], row["TgtEntity"], cands))
    return rows


# --------------------------------------------------------------------------- global matching


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    n_pred: int
    n_ref: int

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}precision": self.precision,
            f"{prefix}recall": self.recall,
            f"{prefix}f1": self.f1,
            f"{prefix}tp": self.tp,
            f"{prefix}n_pred": self.n_pred,
            f"{prefix}n_ref": self.n_ref,
        }


def _prf(tp: int, n_pred: int, n_ref: int) -> PRF:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(p, r, f, tp, n_pred, n_ref)


def global_matching(
    predictions: Iterable[Pair], references: Iterable[Pair], null_references: Iterable[Pair] = ()
) -> PRF:
    """Precision / recall / F1 of predicted mappings, ignoring predictions in ``null_references``."""
    refs = set(references)
    nulls = set(null_references) - refs
    preds = set(predictions) - nulls
    return _prf(len(preds & refs), len(preds), len(refs))


def threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive grid, rounded to avoid float drift (0.30, 0.31, ..., 1.00)."""
    n = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(n), 10)


def percentile_grid(scores: Sequence[float], n: int = 101, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """``n`` evenly spaced thresholds between the ``lo``-th and ``hi``-th percentile (CSLS scores)."""
    scores = np.asarray(list(scores), dtype=float)
    if scores.size == 0:
        return np.array([0.0])
    a, b = np.percentile(scores, [lo, hi])
    return np.linspace(a, b, n)


def threshold_curve(
    scored: dict[Pair, float],
    references: Iterable[Pair],
    null_references: Iterable[Pair],
    grid: Sequence[float],
) -> list[dict[str, float]]:
    """P/R/F1 at every threshold of ``grid`` (keep mappings with score >= θ)."""
    refs = set(references)
    nulls = set(null_references) - refs
    items = [(score, pair in refs) for pair, score in scored.items() if pair not in nulls]
    scores = np.array([s for s, _ in items], dtype=float)
    hits = np.array([h for _, h in items], dtype=bool)
    curve = []
    for theta in grid:
        keep = scores >= theta - 1e-12
        m = _prf(int(hits[keep].sum()), int(keep.sum()), len(refs))
        curve.append({"threshold": float(theta), "precision": m.precision, "recall": m.recall, "f1": m.f1})
    return curve


def select_threshold(curve: list[dict[str, float]]) -> tuple[float, dict[str, float]]:
    """Best-F1 threshold of a **validation** curve. Ties go to the higher threshold.

    Logs a warning when the chosen θ is the first or last grid point (the grid may be too narrow).
    """
    if not curve:
        raise ValueError("empty threshold curve")
    best_i = max(range(len(curve)), key=lambda i: (curve[i]["f1"], curve[i]["threshold"]))
    if best_i in (0, len(curve) - 1) and len(curve) > 1:
        log.warning("chosen threshold %.4f is on the edge of the grid", curve[best_i]["threshold"])
    return curve[best_i]["threshold"], curve[best_i]


def apply_threshold(scored: dict[Pair, float], theta: float) -> dict[Pair, float]:
    return {pair: s for pair, s in scored.items() if s >= theta - 1e-12}


# --------------------------------------------------------------------------- ranking


def rank_of_gold(gold_score: float, other_scores: Iterable[float]) -> int:
    """1-based rank of the gold target among candidates; ties are pessimistic."""
    return 1 + sum(1 for s in other_scores if s >= gold_score)


def rank_from_scores(candidates: Sequence[str], scores: Sequence[float], gold: str) -> int:
    """Rank of ``gold`` given aligned candidate/score lists (NaN scores count as -inf)."""
    clean = [(-math.inf if (s is None or math.isnan(s)) else float(s)) for s in scores]
    gold_score = None
    others = []
    for cand, s in zip(candidates, clean):
        if cand == gold and gold_score is None:
            gold_score = s
        else:
            others.append(s)
    if gold_score is None:
        raise ValueError(f"gold {gold} not among candidates")
    return rank_of_gold(gold_score, others)


def ranking_metrics(ranks: Iterable[int], ks: Sequence[int] = (1, 5, 10), prefix: str = "") -> dict[str, float]:
    """MRR and Hits@k (= Recall@k for single-gold retrieval) from 1-based ranks."""
    ranks = np.asarray(list(ranks), dtype=float)
    if ranks.size == 0:
        return {f"{prefix}mrr": 0.0, **{f"{prefix}hits@{k}": 0.0 for k in ks}, f"{prefix}n": 0}
    out = {f"{prefix}mrr": float(np.mean(1.0 / ranks))}
    for k in ks:
        out[f"{prefix}hits@{k}"] = float(np.mean(ranks <= k))
    out[f"{prefix}n"] = int(ranks.size)
    return out


# --------------------------------------------------------------------------- embedding space


def alignment(x: np.ndarray, y: np.ndarray, alpha: float = 2.0) -> float:
    """Wang & Isola (2020): E ||f(x) - f(y)||^alpha over positive pairs (rows aligned). Lower is better."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(np.mean(np.linalg.norm(x - y, axis=1) ** alpha))


def uniformity(z: np.ndarray, t: float = 2.0) -> float:
    """Wang & Isola (2020): log E exp(-t ||f(x) - f(y)||^2) over distinct pairs. Lower is better."""
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    if n < 2:
        return 0.0
    sq = np.sum(z * z, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * z @ z.T, 0.0)
    iu = np.triu_indices(n, k=1)
    vals = -t * d2[iu]
    m = vals.max()
    return float(m + np.log(np.mean(np.exp(vals - m))))  # stable log-mean-exp


# --------------------------------------------------------------------------- bootstrap


def bootstrap_prf(
    predictions: Iterable[Pair],
    references: Iterable[Pair],
    null_references: Iterable[Pair] = (),
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap CI for F1 (and P, R), resampling **source entities**.

    Each source appearing in the references or in the (non-null) predictions is one unit
    carrying its own TP / #pred / #ref counts. Mappings are almost 1-to-1, so this is close to
    resampling test mappings, while still accounting for false positives.
    """
    refs = set(references)
    nulls = set(null_references) - refs
    preds = set(predictions) - nulls
    units: dict[str, list[int]] = {}
    for s, t in refs:
        units.setdefault(s, [0, 0, 0])[2] += 1
    for s, t in preds:
        u = units.setdefault(s, [0, 0, 0])
        u[1] += 1
        u[0] += int((s, t) in refs)
    counts = np.array(list(units.values()), dtype=np.int64).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    f1s, ps, rs = [], [], []
    for _ in range(n_resamples):
        tp, npred, nref = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        m = _prf(int(tp), int(npred), int(nref))
        f1s.append(m.f1), ps.append(m.precision), rs.append(m.recall)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    point = _prf(int(counts[:, 0].sum()), int(counts[:, 1].sum()), int(counts[:, 2].sum()))
    return {
        "f1": point.f1,
        "f1_ci_low": float(np.percentile(f1s, lo)),
        "f1_ci_high": float(np.percentile(f1s, hi)),
        "precision_ci_low": float(np.percentile(ps, lo)),
        "precision_ci_high": float(np.percentile(ps, hi)),
        "recall_ci_low": float(np.percentile(rs, lo)),
        "recall_ci_high": float(np.percentile(rs, hi)),
    }


def bootstrap_mean(values: Iterable[float], n_resamples: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict[str, float]:
    """Bootstrap CI of a mean, e.g. MRR from per-query reciprocal ranks."""
    v = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_resamples)] if len(v) else [0.0]
    return {
        "mean": float(v.mean()) if len(v) else 0.0,
        "ci_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
    }


# --------------------------------------------------------------------------- results table


def append_summary(row: dict, path: str | Path) -> None:
    """Append one run to ``summary.csv``. New columns are added; old rows get blanks for them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    columns: list[str] = []
    if path.exists():
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
    for key in row:
        if key not in columns:
            columns.append(key)
    rows.append({k: row.get(k, "") for k in columns})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
