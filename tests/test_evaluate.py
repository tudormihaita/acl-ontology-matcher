"""Evaluation harness checks with hand-computed expected values (SPEC §5.3)."""

import logging
import math
from pathlib import Path

import numpy as np
import pytest

from contrastive_om.evaluate import (
    alignment,
    append_summary,
    apply_threshold,
    bootstrap_mean,
    bootstrap_prf,
    global_matching,
    percentile_grid,
    rank_from_scores,
    ranking_metrics,
    read_candidates,
    read_mappings,
    read_scored_mappings,
    select_threshold,
    threshold_curve,
    threshold_grid,
    uniformity,
    write_scored_mappings,
)

# test split (4 refs) and validation split (2 refs, null references when scoring test)
TEST = [("a1", "b1"), ("a2", "b2"), ("a3", "b3"), ("a4", "b4")]
VAL = [("a5", "b5"), ("a6", "b6")]
PRED = [
    ("a1", "b1"),  # TP
    ("a2", "b2"),  # TP
    ("a3", "bX"),  # FP (wrong target)
    ("a5", "b5"),  # exactly a validation mapping -> ignored
    ("a6", "b7"),  # source is in validation, but not the exact pair -> FP
    ("a7", "b8"),  # FP (unmapped source)
]


# --------------------------------------------------------------------------- global matching


def test_global_matching_with_null_references():
    m = global_matching(PRED, TEST, null_references=VAL)
    # 5 counted predictions, 2 TP, 4 refs: P = 2/5, R = 2/4, F1 = 2PR/(P+R) = 0.4/0.9 = 4/9
    assert (m.tp, m.n_pred, m.n_ref) == (2, 5, 4)
    assert m.precision == pytest.approx(0.4)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(4 / 9)


def test_global_matching_without_null_references_counts_val_hit_as_fp():
    m = global_matching(PRED, TEST)
    # (a5, b5) now counts as FP: 6 predictions, 2 TP -> P = 1/3, R = 1/2, F1 = 0.4
    assert (m.tp, m.n_pred) == (2, 6)
    assert m.f1 == pytest.approx(0.4)


def test_null_references_are_symmetric():
    # scoring validation with test as null refs: (a1,b1),(a2,b2) ignored; (a5,b5) TP;
    # counted preds: (a3,bX), (a5,b5), (a6,b7), (a7,b8) -> P = 1/4, R = 1/2, F1 = 1/3
    m = global_matching(PRED, VAL, null_references=TEST)
    assert (m.tp, m.n_pred, m.n_ref) == (1, 4, 2)
    assert m.f1 == pytest.approx(1 / 3)


def test_null_reference_never_removes_a_true_reference():
    m = global_matching([("a1", "b1")], TEST, null_references=[("a1", "b1")])
    assert m.tp == 1 and m.n_pred == 1


def test_global_matching_edge_cases():
    assert global_matching([], TEST).f1 == 0.0
    assert global_matching([], TEST).precision == 0.0
    perfect = global_matching(TEST, TEST, VAL)
    assert (perfect.precision, perfect.recall, perfect.f1) == (1.0, 1.0, 1.0)
    assert global_matching(TEST + TEST, TEST).n_pred == 4  # duplicates collapse


# --------------------------------------------------------------------------- thresholds

SCORED = {
    ("a1", "b1"): 0.9,
    ("a2", "b2"): 0.7,
    ("a3", "bX"): 0.8,
    ("a5", "b5"): 0.95,  # validation pair: ignored on test
    ("a7", "b8"): 0.6,
}


def test_threshold_grid_is_inclusive_and_exact():
    g = threshold_grid(0.30, 1.00, 0.01)
    assert len(g) == 71
    assert g[0] == 0.30 and g[-1] == 1.00
    assert 0.57 in g  # no float drift


def test_threshold_curve_hand_computed():
    curve = threshold_curve(SCORED, TEST, VAL, [0.6, 0.7, 0.8, 0.9, 1.0])
    f1 = [round(c["f1"], 6) for c in curve]
    # θ=.6: 4 preds, 2 TP -> P=.5  R=.5  F1=.5
    # θ=.7: 3 preds, 2 TP -> P=2/3 R=.5  F1=4/7
    # θ=.8: 2 preds, 1 TP -> P=.5  R=.25 F1=1/3
    # θ=.9: 1 pred,  1 TP -> P=1   R=.25 F1=.4
    # θ=1.: 0 preds       -> 0
    assert f1 == [0.5, round(4 / 7, 6), round(1 / 3, 6), 0.4, 0.0]
    assert curve[1]["precision"] == pytest.approx(2 / 3)
    theta, best = select_threshold(curve)
    assert theta == 0.7 and best["f1"] == pytest.approx(4 / 7)


def test_threshold_is_inclusive():
    assert apply_threshold(SCORED, 0.7) == {k: v for k, v in SCORED.items() if v >= 0.7}
    assert ("a2", "b2") in apply_threshold(SCORED, 0.7)


def test_select_threshold_ties_go_to_higher_threshold():
    curve = [{"threshold": 0.5, "f1": 0.6}, {"threshold": 0.6, "f1": 0.8}, {"threshold": 0.7, "f1": 0.8},
             {"threshold": 0.8, "f1": 0.1}]
    assert select_threshold(curve)[0] == 0.7


def test_select_threshold_warns_on_grid_edge(caplog):
    curve = threshold_curve({("a1", "b1"): 1.0}, TEST, VAL, [0.5, 0.75, 1.0])
    with caplog.at_level(logging.WARNING):
        theta, _ = select_threshold(curve)
    assert theta == 1.0  # ties (all F1 = 0.4) -> highest threshold, which is the edge
    assert "edge of the grid" in caplog.text


def test_percentile_grid():
    g = percentile_grid(np.arange(101), n=3)  # 1st/99th percentile of 0..100 = 1, 99
    assert np.allclose(g, [1.0, 50.0, 99.0])


# --------------------------------------------------------------------------- ranking


def test_rank_ties_are_pessimistic():
    # gold 0.5; others 0.9 (higher), 0.5 (tie), 0.1 -> rank 3
    assert rank_from_scores(["g", "x", "y", "z"], [0.5, 0.9, 0.5, 0.1], "g") == 3


def test_rank_nan_scores_count_as_minus_infinity():
    assert rank_from_scores(["x", "g", "y"], [float("nan"), 0.2, 0.1], "g") == 1
    assert rank_from_scores(["x", "g"], [0.1, float("nan")], "g") == 2


def test_rank_requires_gold():
    with pytest.raises(ValueError):
        rank_from_scores(["x"], [0.1], "g")


def test_ranking_metrics_hand_computed():
    m = ranking_metrics([1, 2, 4, 10, 20], ks=(1, 5, 10))
    # MRR = (1 + 1/2 + 1/4 + 1/10 + 1/20) / 5 = 1.9 / 5
    assert m["mrr"] == pytest.approx(0.38)
    assert m["hits@1"] == pytest.approx(0.2)
    assert m["hits@5"] == pytest.approx(0.6)
    assert m["hits@10"] == pytest.approx(0.8)
    assert m["n"] == 5


def test_ranking_metrics_empty_and_prefix():
    assert ranking_metrics([], prefix="val_")["val_mrr"] == 0.0
    assert "full_hits@1" in ranking_metrics([1], prefix="full_")


# --------------------------------------------------------------------------- embedding space


def test_alignment():
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[1.0, 0.0], [-1.0, 0.0]])
    # squared distances 0 and 2 -> mean 1
    assert alignment(x, y) == pytest.approx(1.0)


def test_uniformity():
    two = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert uniformity(two) == pytest.approx(-8.0)  # one pair, d^2 = 4 -> log exp(-8)
    three = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    expected = math.log((2 * math.exp(-4) + math.exp(-8)) / 3)  # d^2 = 2, 4, 2
    assert uniformity(three) == pytest.approx(expected)


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_prf_point_and_ci():
    out = bootstrap_prf(PRED, TEST, VAL, n_resamples=500, seed=1)
    assert out["f1"] == pytest.approx(4 / 9)  # same point estimate as global_matching
    assert out["f1_ci_low"] <= out["f1"] <= out["f1_ci_high"]
    assert out == bootstrap_prf(PRED, TEST, VAL, n_resamples=500, seed=1)  # deterministic


def test_bootstrap_independent_of_input_order():
    a = bootstrap_prf(PRED, TEST, VAL, n_resamples=200, seed=3)
    b = bootstrap_prf(list(reversed(PRED)), list(reversed(TEST)), VAL, n_resamples=200, seed=3)
    assert a == b


def test_bootstrap_perfect_predictions():
    out = bootstrap_prf(TEST, TEST, n_resamples=200)
    assert out["f1_ci_low"] == out["f1_ci_high"] == 1.0


def test_bootstrap_mean():
    out = bootstrap_mean([1.0, 0.5, 0.0, 0.5], n_resamples=300)
    assert out["mean"] == pytest.approx(0.5)
    assert 0.0 <= out["ci_low"] <= 0.5 <= out["ci_high"] <= 1.0


# --------------------------------------------------------------------------- I/O


def test_read_candidates(tmp_path: Path):
    p = tmp_path / "cands.tsv"
    p.write_text("SrcEntity\tTgtEntity\tTgtCandidates\na1\tb1\t('b9', 'b1', 'b2')\n")
    rows = read_candidates(p)
    assert rows[0].src == "a1" and rows[0].gold == "b1"
    assert rows[0].candidates == ("b9", "b1", "b2")


def test_read_candidates_requires_gold(tmp_path: Path):
    p = tmp_path / "cands.tsv"
    p.write_text("SrcEntity\tTgtEntity\tTgtCandidates\na1\tb1\t('b9', 'b2')\n")
    with pytest.raises(ValueError):
        read_candidates(p)


def test_mappings_roundtrip(tmp_path: Path):
    p = tmp_path / "m.tsv"
    write_scored_mappings(SCORED, p)
    back = read_scored_mappings(p)
    assert back.keys() == SCORED.keys()
    assert all(back[k] == pytest.approx(v) for k, v in SCORED.items())
    assert read_mappings(p)[0] == ("a5", "b5")  # sorted by descending score


def test_append_summary_adds_columns(tmp_path: Path):
    p = tmp_path / "summary.csv"
    append_summary({"name": "e0", "f1": 0.5}, p)
    append_summary({"name": "e1", "f1": 0.6, "mrr": 0.7}, p)
    lines = p.read_text().splitlines()
    assert lines[0] == "name,f1,mrr"
    assert lines[1] == "e0,0.5,"
    assert lines[2] == "e1,0.6,0.7"


# --------------------------------------------------------------------------- real splits

DATA = Path(__file__).parents[1] / "data/raw/bio-ml/omim-ordo/refs_equiv"


@pytest.mark.skipif(not DATA.exists(), reason="Bio-ML data not downloaded")
def test_real_splits_are_consistent():
    val, test, full = (set(read_mappings(DATA / f)) for f in ("train.tsv", "test.tsv", "full.tsv"))
    assert (len(val), len(test)) == (1116, 2605)
    assert not val & test and val | test == full
    cands = read_candidates(DATA / "test.cands.tsv")
    assert {(r.src, r.gold) for r in cands} == test
    assert all(len(r.candidates) == 101 for r in cands)
