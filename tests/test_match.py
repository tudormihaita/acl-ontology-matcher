"""Matching on toy matrices with hand-computed answers (SPEC §9, Phase 4)."""

import numpy as np
import pytest

from contrastive_om.match import (
    class_scores,
    csls,
    jaccard_class_scores,
    mask_unalignable,
    neighbor_rescore,
    select_argmax,
    select_mutual,
    select_stable_marriage,
    string_match,
    topk,
)
from contrastive_om.represent import ClassEmbeddings, l2_normalize


def emb(groups: list[list[list[float]]], names=None) -> ClassEmbeddings:
    """ClassEmbeddings from per-class lists of label vectors."""
    labels, vecs, offsets = [], [], []
    for ci, group in enumerate(groups):
        offsets.append(len(labels))
        for li, v in enumerate(group):
            labels.append(f"c{ci}-l{li}")
            vecs.append(v)
    vecs = l2_normalize(np.asarray(vecs, np.float32))
    offsets = np.asarray(offsets, np.int64)
    class_vecs = l2_normalize(np.add.reduceat(vecs, offsets, axis=0))
    return ClassEmbeddings(names or [f"c{i}" for i in range(len(groups))], labels, vecs, offsets, class_vecs)


def brute_max(src: ClassEmbeddings, tgt: ClassEmbeddings) -> np.ndarray:
    out = np.zeros((len(src.iris), len(tgt.iris)), np.float32)
    for i in range(len(src.iris)):
        for j in range(len(tgt.iris)):
            out[i, j] = (src.label_vecs[src.label_slice(i)] @ tgt.label_vecs[tgt.label_slice(j)].T).max()
    return out


# --------------------------------------------------------------------------- scores


def test_max_and_mean_class_scores():
    src = emb([[[1, 0], [0, 1]], [[1, 1]]])  # class 0 has two labels
    tgt = emb([[[0, 1]], [[1, 0], [-1, 0]]])
    m = class_scores(src, tgt, "max")
    # class 0 contains [0,1] -> exact match with target 0; and [1,0] -> exact match with target 1
    assert m[0, 0] == pytest.approx(1.0) and m[0, 1] == pytest.approx(1.0)
    assert m[1, 0] == pytest.approx(np.sqrt(0.5))
    mean = class_scores(src, tgt, "mean")
    # mean of [1,0],[0,1] -> [1,1]/sqrt2 ; target 0 = [0,1] -> cos = 1/sqrt2
    assert mean[0, 0] == pytest.approx(np.sqrt(0.5))


@pytest.mark.parametrize("chunk", [1, 2, 3, 100])
def test_chunked_max_equals_brute_force(chunk):
    rng = np.random.default_rng(0)
    src = emb([[rng.normal(size=8) for _ in range(rng.integers(1, 4))] for _ in range(7)])
    tgt = emb([[rng.normal(size=8) for _ in range(rng.integers(1, 4))] for _ in range(5)])
    assert np.allclose(class_scores(src, tgt, "max", chunk=chunk), brute_max(src, tgt), atol=1e-6)


def test_mask_unalignable():
    s = mask_unalignable(np.ones((2, 3), np.float32), np.array([True, False]), np.array([True, True, False]))
    assert np.isinf(s[1]).all() and np.isinf(s[:, 2]).all() and s[0, 0] == 1


def test_csls_hand_computed_and_hubness():
    s = np.array([[0.9, 0.8], [0.85, 0.84]], np.float32)
    c = csls(s, k=1)
    # r_T = row maxima [0.9, 0.85]; r_S = column maxima [0.9, 0.84]
    expected = 2 * s - np.array([0.9, 0.85])[:, None] - np.array([0.9, 0.84])[None, :]
    assert np.allclose(c, expected, atol=1e-6)
    # cosine: both sources pick the "hub" target 0; CSLS moves source 1 to target 1
    assert s[1].argmax() == 0 and c[1].argmax() == 1


def test_csls_ignores_masked_entries():
    s = np.array([[0.9, -np.inf], [0.5, 0.4]], np.float32)
    c = csls(s, k=2)
    assert np.isinf(c[0, 1]) and np.isfinite(c[1]).all()


def test_topk_sorted():
    idx, vals = topk(np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.7]], np.float32), 2)
    assert idx.tolist() == [[1, 2], [2, 0]]
    assert np.allclose(vals, [[0.9, 0.5], [0.7, 0.3]])


# --------------------------------------------------------------------------- selection


def test_argmax_and_mutual():
    s = np.array([[0.9, 0.1], [0.8, 0.2]], np.float32)
    idx, vals = topk(s, 2)
    assert select_argmax(idx, vals) == {(0, 0): pytest.approx(0.9), (1, 0): pytest.approx(0.8)}
    # target 0's best source is 0, so (1, 0) is not mutual
    assert set(select_mutual(idx, vals)) == {(0, 0)}


def test_selection_skips_rows_without_candidates():
    s = np.array([[-np.inf, -np.inf], [0.3, 0.2]], np.float32)
    idx, vals = topk(s, 2)
    assert set(select_argmax(idx, vals)) == {(1, 0)}
    assert set(select_stable_marriage(idx, vals)) == {(1, 0)}


def test_stable_marriage_hand_computed():
    # both sources prefer target 0; source 0 wins it (0.9 > 0.8), source 1 falls back to target 1
    s = np.array([[0.9, 0.1], [0.8, 0.2]], np.float32)
    assert select_stable_marriage(*topk(s, 2)) == {(0, 0): pytest.approx(0.9), (1, 1): pytest.approx(0.2)}
    # a later, better proposer displaces an earlier one
    s = np.array([[0.8, 0.7], [0.9, 0.1]], np.float32)
    assert set(select_stable_marriage(*topk(s, 2))) == {(1, 0), (0, 1)}


def test_stable_marriage_is_one_to_one_and_stable():
    rng = np.random.default_rng(3)
    s = rng.random((30, 20)).astype(np.float32)
    idx, vals = topk(s, 5)
    m = select_stable_marriage(idx, vals)
    targets = [t for _, t in m]
    assert len(targets) == len(set(targets))
    partner_score = {t: v for (_, t), v in m.items()}
    my_score = {i: v for (i, _), v in m.items()}
    for i in range(idx.shape[0]):  # no blocking pair inside the top-k lists
        for j in range(idx.shape[1]):
            t, v = int(idx[i, j]), float(vals[i, j])
            prefers = i not in my_score or v > my_score[i]
            accepted = t not in partner_score or v > partner_score[t]
            assert not (prefers and accepted), f"blocking pair ({i}, {t})"


# --------------------------------------------------------------------------- neighbor re-scoring


def test_neighbor_rescore_hand_computed():
    s = np.array([[0.8, 0.8, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.1, 0.9]], np.float32)
    src_parents = [[2], [], []]  # source 0 is a child of source 2
    tgt_parents = [[2], [], []]  # target 0 is a child of target 2; target 1 has no parent
    idx, vals = np.array([[1, 0]]), np.array([[0.8, 0.8]], np.float32)
    new_idx, new_vals = neighbor_rescore(s, idx, vals, src_parents[:1], tgt_parents, lam=0.5)
    # (s0, t0): 0.5 * 0.8 + 0.5 * s[2, 2] = 0.85 ; (s0, t1): no target parent -> unchanged 0.8
    assert new_idx.tolist() == [[0, 1]]
    assert np.allclose(new_vals, [[0.85, 0.8]])


def test_neighbor_rescore_off():
    idx, vals = np.array([[1, 0]]), np.array([[0.8, 0.7]], np.float32)
    assert neighbor_rescore(np.zeros((1, 2)), idx, vals, [[0]], [[0], [0]], 0.0) == (idx, vals)


# --------------------------------------------------------------------------- baseline


def test_string_match():
    src = {"a": ["lung disease", "pulmonary disease"], "b": ["heart disease"]}
    tgt = {"x": ["lung disease"], "y": ["cardiac disease"], "z": ["pulmonary disease"]}
    assert string_match(src, tgt) == {("a", "x"): 1.0, ("a", "z"): 1.0}


def test_jaccard_class_scores():
    def layout(groups):
        labels = [lab for g in groups for lab in g]
        offsets = np.cumsum([0] + [len(g) for g in groups[:-1]]).astype(np.int64)
        empty = np.zeros((0, 0), np.float32)
        return ClassEmbeddings([f"c{i}" for i in range(len(groups))], labels, empty, offsets, empty)

    src = layout([["dilated cardiomyopathy 1a", "cmd1a"], ["lung disease"]])
    tgt = layout([["dilated cardiomyopathy type 1a"], ["pulmonary disease", "lung disease"]])
    j = jaccard_class_scores(src, tgt)
    assert j[0, 0] == pytest.approx(3 / 4)  # {dilated, cardiomyopathy, 1a} / 4 tokens
    assert j[1, 1] == pytest.approx(1.0)  # best label pair is exact
    assert j[0, 1] == 0.0
    assert np.allclose(jaccard_class_scores(src, tgt, rows=[1]), j[[1]])
