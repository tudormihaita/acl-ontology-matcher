"""Matching from the embedding space (SPEC §4.5) and the string-match baseline (SPEC §4.6).

Pipeline: class-pair scores (``max`` over label pairs, or ``mean`` class vectors), optional CSLS,
top-k candidates per source, optional neighbor-aware re-scoring of those candidates, selection
(``argmax`` | ``mutual`` | ``stable_marriage``). Thresholding happens in ``pipeline.py`` with the
θ chosen on validation. No candidate index, no extension, no repair.

The class x class score matrix is kept in memory (about 9.6k x 9.3k float32 for OMIM-ORDO);
the label x label matrix is never built: label scores are reduced to class scores in chunks.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from contrastive_om.represent import ClassEmbeddings

log = logging.getLogger(__name__)
NEG_INF = np.float32(-np.inf)


# --------------------------------------------------------------------------- scores


def class_scores(src: ClassEmbeddings, tgt: ClassEmbeddings, class_repr: str = "max", chunk: int = 512) -> np.ndarray:
    """(n_src_classes, n_tgt_classes) cosine scores."""
    if class_repr == "mean":
        return (src.class_vecs @ tgt.class_vecs.T).astype(np.float32)
    if class_repr != "max":
        raise ValueError(f"unknown class_repr '{class_repr}'")
    out = np.empty((len(src.iris), len(tgt.iris)), dtype=np.float32)
    for start in range(0, len(src.iris), chunk):
        stop = min(start + chunk, len(src.iris))
        lab_start = int(src.offsets[start])
        lab_stop = int(src.offsets[stop]) if stop < len(src.iris) else len(src.labels)
        block = src.label_vecs[lab_start:lab_stop] @ tgt.label_vecs.T  # (labels in chunk, tgt labels)
        block = np.maximum.reduceat(block, tgt.offsets, axis=1)  # -> (labels in chunk, tgt classes)
        out[start:stop] = np.maximum.reduceat(block, src.offsets[start:stop] - lab_start, axis=0)
    return out


def mask_unalignable(scores: np.ndarray, src_ok: np.ndarray, tgt_ok: np.ndarray) -> np.ndarray:
    """Set rows/columns of classes that may not be matched (``use_in_alignment=false``) to -inf."""
    scores = scores.copy()
    scores[~src_ok, :] = NEG_INF
    scores[:, ~tgt_ok] = NEG_INF
    return scores


def _topk_mean(m: np.ndarray, k: int, axis: int, chunk: int = 2048) -> np.ndarray:
    """Mean of the k largest finite values along ``axis`` (rows: axis=1, columns: axis=0)."""
    m = m if axis == 1 else m.T
    k = max(1, min(k, m.shape[1]))
    out = np.empty(m.shape[0], dtype=np.float32)
    for start in range(0, m.shape[0], chunk):
        block = m[start:start + chunk]
        top = -np.partition(-block, k - 1, axis=1)[:, :k]
        top = np.where(np.isfinite(top), top, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # fully masked rows: mean of nothing -> 0
            out[start:start + chunk] = np.nan_to_num(np.nanmean(top, axis=1), nan=0.0)
    return out


def csls(scores: np.ndarray, k: int = 10) -> np.ndarray:
    """Cross-domain similarity local scaling (Conneau et al., 2018):
    ``csls(s,t) = 2 cos(s,t) - r_T(s) - r_S(t)``, r = mean cosine to the k nearest neighbors on
    the other side. Penalizes "hub" targets that are close to everything."""
    r_t = _topk_mean(scores, k, axis=1)  # per source
    r_s = _topk_mean(scores, k, axis=0)  # per target
    with np.errstate(invalid="ignore"):
        out = 2 * scores - r_t[:, None] - r_s[None, :]
    return out.astype(np.float32)


def topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the k best targets per source, best first."""
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(vals, order, axis=1)


# --------------------------------------------------------------------------- neighbor re-scoring


def neighbor_rescore(
    scores: np.ndarray,
    idx: np.ndarray,
    vals: np.ndarray,
    src_parents: list[list[int]],
    tgt_parents: list[list[int]],
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``score' = (1-λ) score + λ sim_N`` for the top-k candidates (SPEC §4.5 step 5).

    ``sim_N(s,t)`` = mean over parents p of s of max over parents q of t of ``scores[p, q]``.
    If either side has no (informative) parent, ``sim_N = score``: no penalty.
    Candidates are re-sorted by the new score.
    """
    if lam <= 0:
        return idx, vals
    new = vals.copy()
    for i in range(idx.shape[0]):
        ps = src_parents[i]
        if not ps:
            continue
        for j in range(idx.shape[1]):
            if not np.isfinite(vals[i, j]):
                continue
            qs = tgt_parents[idx[i, j]]
            if not qs:
                continue
            sim_n = float(np.mean(scores[np.ix_(ps, qs)].max(axis=1)))
            if np.isfinite(sim_n):
                new[i, j] = (1 - lam) * vals[i, j] + lam * sim_n
    order = np.argsort(-new, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(new, order, axis=1)


# --------------------------------------------------------------------------- selection


def select_argmax(idx: np.ndarray, vals: np.ndarray) -> dict[tuple[int, int], float]:
    return {(i, int(idx[i, 0])): float(vals[i, 0]) for i in range(idx.shape[0]) if np.isfinite(vals[i, 0])}


def select_mutual(idx: np.ndarray, vals: np.ndarray) -> dict[tuple[int, int], float]:
    """Keep (s, t) if t is s's best target and s is t's best source.

    "t's best source" is taken over the candidate graph (sources having t in their top-k),
    which is exact whenever t's true best source ranks t within its top-k.
    """
    best_src: dict[int, tuple[float, int]] = {}
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                t = int(idx[i, j])
                if t not in best_src or v > best_src[t][0]:
                    best_src[t] = (float(v), i)
    out = {}
    for (i, t), v in select_argmax(idx, vals).items():
        if best_src.get(t, (None, -1))[1] == i:
            out[(i, t)] = v
    return out


def select_stable_marriage(idx: np.ndarray, vals: np.ndarray) -> dict[tuple[int, int], float]:
    """One-to-one Gale-Shapley over the top-k lists (sources propose in score order; a target
    keeps the proposer with the higher score). Sources may stay unmatched."""
    n = idx.shape[0]
    next_choice = [0] * n
    engaged_to: dict[int, tuple[int, float]] = {}  # target -> (source, score)
    free = [i for i in range(n) if np.isfinite(vals[i, 0])]
    while free:
        i = free.pop()
        while next_choice[i] < idx.shape[1]:
            j = next_choice[i]
            next_choice[i] += 1
            v = float(vals[i, j])
            if not np.isfinite(v):
                break
            t = int(idx[i, j])
            if t not in engaged_to:
                engaged_to[t] = (i, v)
                break
            other, ov = engaged_to[t]
            if v > ov or (v == ov and i < other):  # deterministic tie-break
                engaged_to[t] = (i, v)
                free.append(other)
                break
    return {(i, t): v for t, (i, v) in engaged_to.items()}


SELECTIONS = {"argmax": select_argmax, "mutual": select_mutual, "stable_marriage": select_stable_marriage}


# --------------------------------------------------------------------------- string-match baseline

_WORD = re.compile(r"[a-z0-9]+")


def _token_matrix(labels: list[str], vocab: dict[str, int]) -> sp.csr_matrix:
    rows, cols = [], []
    for r, lab in enumerate(labels):
        for tok in set(_WORD.findall(lab)):
            if tok in vocab:
                rows.append(r)
                cols.append(vocab[tok])
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(labels), len(vocab)))


def jaccard_class_scores(src: ClassEmbeddings, tgt: ClassEmbeddings, rows: list[int] | None = None,
                         chunk: int = 256) -> np.ndarray:
    """Best token-Jaccard over label pairs, for the given source class rows (baseline ranking score).

    Only ``src.labels``/``offsets`` and ``tgt.labels``/``offsets`` are used (no vectors needed).
    """
    rows = list(range(len(src.iris))) if rows is None else rows
    vocab: dict[str, int] = {}
    for lab in src.labels + tgt.labels:
        for tok in _WORD.findall(lab):
            vocab.setdefault(tok, len(vocab))
    tgt_m = _token_matrix(tgt.labels, vocab)
    tgt_size = np.asarray(tgt_m.sum(axis=1)).ravel()
    out = np.zeros((len(rows), len(tgt.iris)), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        slices = [src.label_slice(i) for i in part]
        labs = [lab for s in slices for lab in src.labels[s]]
        owners = np.repeat(np.arange(len(part)), [s.stop - s.start for s in slices])
        src_m = _token_matrix(labs, vocab)
        inter = (src_m @ tgt_m.T).toarray()
        src_size = np.asarray(src_m.sum(axis=1)).ravel()
        union = src_size[:, None] + tgt_size[None, :] - inter
        jac = np.where(union > 0, inter / np.maximum(union, 1), 0.0).astype(np.float32)
        jac = np.maximum.reduceat(jac, tgt.offsets, axis=1)
        offsets = np.searchsorted(owners, np.arange(len(part)))
        out[start:start + len(part)] = np.maximum.reduceat(jac, offsets, axis=0)
    return out


def string_match(src_labels: dict[str, list[str]], tgt_labels: dict[str, list[str]]) -> dict[tuple[str, str], float]:
    """All (s, t) sharing at least one normalized label, score 1.0 (SPEC §4.6)."""
    index: dict[str, set[str]] = {}
    for t, labs in tgt_labels.items():
        for lab in labs:
            index.setdefault(lab, set()).add(t)
    out = {}
    for s, labs in src_labels.items():
        for lab in labs:
            for t in index.get(lab, ()):
                out[(s, t)] = 1.0
    return out


# --------------------------------------------------------------------------- high level


@dataclass
class MatchResult:
    scores: np.ndarray  # final scoring matrix (cosine or CSLS), unalignable masked to -inf
    candidates: tuple[np.ndarray, np.ndarray]  # top-k (idx, vals) after neighbor re-scoring
    mappings: dict[tuple[str, str], float]  # selected (src_iri, tgt_iri) -> score, before θ


def match(
    src: ClassEmbeddings,
    tgt: ClassEmbeddings,
    src_ok: np.ndarray,
    tgt_ok: np.ndarray,
    cfg_match,
    src_parents: list[list[int]] | None = None,
    tgt_parents: list[list[int]] | None = None,
) -> MatchResult:
    scores = mask_unalignable(class_scores(src, tgt, cfg_match.class_repr), src_ok, tgt_ok)
    if cfg_match.scoring == "csls":
        scores = csls(scores, int(cfg_match.csls_k))
    elif cfg_match.scoring != "cosine":
        raise ValueError(f"unknown scoring '{cfg_match.scoring}'")
    idx, vals = topk(scores, int(cfg_match.top_k))
    lam = float(cfg_match.get("neighbor_lambda", 0.0) or 0.0)
    if lam > 0:
        idx, vals = neighbor_rescore(scores, idx, vals, src_parents or [], tgt_parents or [], lam)
    selected = SELECTIONS[cfg_match.selection](idx, vals)
    mappings = {(src.iris[i], tgt.iris[t]): v for (i, t), v in selected.items()}
    return MatchResult(scores, (idx, vals), mappings)
