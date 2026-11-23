import random
from pathlib import Path

import numpy as np
import pytest

from contrastive_om.config import Config
from contrastive_om.lexicon import Lexicon, Variant
from contrastive_om.ontology import load_ontology
from contrastive_om.pairs import (
    NegativeSampler,
    PairRow,
    assert_no_reference_leakage,
    load_rows,
    mine_ontology,
    positive_pairs,
    save_rows,
)

FIX = Path(__file__).parent / "fixtures"
A = "http://example.org/a#"
LABELS = ["rdfs:label", "skos:altLabel", "oboInOwl:hasExactSynonym"]


@pytest.fixture(scope="module")
def ont():
    return load_ontology(FIX / "toy_a.owl", LABELS)


def pairs_cfg(**kw):
    base = {"positives": ["syn"], "hard_negatives": [], "num_hard_negatives": 1, "max_pairs_per_class": 10,
            "filter_cos": 0.85}
    return Config({**base, **kw})


def by_class(rows):
    out = {}
    for r in rows:
        out.setdefault(r.class_iri, []).append(r)
    return out


# --------------------------------------------------------------------------- positives


def test_syn_positives_are_all_ordered_label_pairs(ont):
    rows, stats = mine_ontology(ont, pairs_cfg(), None, seed=42)
    dcm = by_class(rows)[A + "DilatedCardiomyopathy1A"]
    labels = ont[A + "DilatedCardiomyopathy1A"].labels  # 3 labels -> 6 ordered pairs
    assert {(r.anchor, r.positive) for r in dcm} == {(a, b) for a in labels for b in labels if a != b}
    assert all(r.source == "syn" for r in rows)
    assert A + "Disease" not in by_class(rows)  # single label: no syn positive
    assert stats["single_label_classes_covered"] == 0


def test_max_pairs_per_class(ont):
    rows, _ = mine_ontology(ont, pairs_cfg(max_pairs_per_class=2), None, seed=42)
    assert max(len(v) for v in by_class(rows).values()) == 2


def test_lex_and_syn_are_interleaved_under_the_cap(ont):
    iri = A + "DilatedCardiomyopathy1A"
    lex = Lexicon("toy_a", variants={iri: [Variant("cmd1a", "cmd 1a", "table")],
                                     A + "Disease": [Variant("disease", "diseases", "rule:x")]})
    pairs = positive_pairs(ont, lex, ["syn", "lex"], max_pairs=4, rng=random.Random(0))
    sources = [src for _, _, c, src in pairs if c == iri]
    assert len(sources) == 4 and sources.count("lex") == 2  # both directions of the variant
    # a single-label class gets positives from the lexicon only
    assert [(a, b) for a, b, c, _ in pairs if c == A + "Disease"] == [("disease", "diseases"), ("diseases", "disease")] \
        or sorted((a, b) for a, b, c, _ in pairs if c == A + "Disease") == [("disease", "diseases"), ("diseases", "disease")]


def test_mining_is_deterministic(ont):
    cfg = pairs_cfg(hard_negatives=["sibling"])
    assert mine_ontology(ont, cfg, None, seed=7) == mine_ontology(ont, cfg, None, seed=7)


# --------------------------------------------------------------------------- negatives


def test_sibling_negatives(ont):
    rows, stats = mine_ontology(ont, pairs_cfg(hard_negatives=["sibling"]), None, seed=42)
    for r in by_class(rows)[A + "DilatedCardiomyopathy1A"]:
        assert r.negative_class_iris == [A + "DilatedCardiomyopathy1B"]
        assert r.negative_kinds == ["sibling"]
        assert r.negatives[0] in ont[A + "DilatedCardiomyopathy1B"].labels
    # Cardiomyopathy has no sibling -> filled with a random negative, and counted
    for r in by_class(rows)[A + "Cardiomyopathy"]:
        assert r.negative_kinds == ["random"]
        assert r.negative_class_iris[0] != A + "Cardiomyopathy"
    assert stats["rows_filled_with_random_negatives"] == len(by_class(rows)[A + "Cardiomyopathy"])


def test_negative_is_never_a_label_of_the_anchor_class(ont):
    for kinds in (["sibling"], ["hierarchy"], []):
        rows, _ = mine_ontology(ont, pairs_cfg(hard_negatives=kinds), None, seed=1)
        for r in rows:
            assert not set(r.negatives) & set(ont[r.class_iri].labels)
            assert r.class_iri not in r.negative_class_iris


def test_no_hard_negatives_means_in_batch_only(ont):
    # E2 "none": no negative columns at all, even though num_hard_negatives defaults to 1
    rows, stats = mine_ontology(ont, pairs_cfg(hard_negatives=[], num_hard_negatives=1), None, seed=1)
    assert all(r.negatives == [] and r.negative_class_iris == [] for r in rows)
    assert stats["rows_filled_with_random_negatives"] == 0


def test_every_row_has_the_same_number_of_negatives(ont):
    rows, _ = mine_ontology(ont, pairs_cfg(hard_negatives=["sibling"], num_hard_negatives=2), None, seed=1)
    assert {len(r.negatives) for r in rows} == {2}


def test_hierarchy_negatives(ont):
    rows, _ = mine_ontology(ont, pairs_cfg(hard_negatives=["hierarchy"]), None, seed=42)
    allowed = {A + "HeartDisease", A + "DilatedCardiomyopathy1A", A + "DilatedCardiomyopathy1B"}
    for r in by_class(rows)[A + "Cardiomyopathy"]:
        assert r.negative_kinds == ["hierarchy"] and r.negative_class_iris[0] in allowed


def test_sibling_filtered_requires_embeddings(ont):
    with pytest.raises(ValueError):
        NegativeSampler(ont, ["sibling_filtered"])
    with pytest.raises(ValueError):
        NegativeSampler(ont, ["siblings"])  # typo


def _embeddings(ont, close: set[tuple[str, str]] = frozenset()):
    """Orthogonal unit vectors per label; pairs in ``close`` share a vector (cosine 1)."""
    labels = sorted({lab for c in ont for lab in c.labels})
    emb = {lab: np.eye(len(labels))[i] for i, lab in enumerate(labels)}
    for a, b in close:
        emb[b] = emb[a]
    return emb


def test_sibling_filtered_cosine_filter(ont):
    iri = A + "DilatedCardiomyopathy1A"
    anchor = "dilated cardiomyopathy 1a"
    sampler = NegativeSampler(ont, ["sibling_filtered"], 0.85, _embeddings(ont))
    # toy_a has a disjointness axiom, so rule (c) applies: 1B is not declared disjoint -> dropped
    assert sampler.sample(iri, anchor, 1, random.Random(0)) == []
    sampler.has_disjointness = False
    assert sampler.sample(iri, anchor, 1, random.Random(0))[0][1] == A + "DilatedCardiomyopathy1B"
    # (b): ONE 1B label too similar to the anchor drops the whole sibling (not just that label;
    # otherwise the sibling would come back through its abbreviation "cmd1b")
    close = {(anchor, "dilated cardiomyopathy 1b")}
    sampler = NegativeSampler(ont, ["sibling_filtered"], 0.85, _embeddings(ont, close))
    sampler.has_disjointness = False
    assert sampler.sample(iri, anchor, 1, random.Random(0)) == []


def test_sibling_filtered_disjointness_keeps_declared_disjoint(ont):
    # HeartDisease's siblings: LungDisease (declared disjoint) and HelperClass (not)
    sampler = NegativeSampler(ont, ["sibling_filtered"], 0.85, _embeddings(ont))
    got = {sampler.sample(A + "HeartDisease", "heart disease", 1, random.Random(s))[0][1] for s in range(20)}
    assert got == {A + "LungDisease"}


# --------------------------------------------------------------------------- leakage and I/O


def _row(cls, negs, ontology="toy_a"):
    return PairRow("a", "b", ["n"] * len(negs), cls, negs, ontology, "syn", ["sibling"] * len(negs))


def test_leakage_assert(ont):
    onts = {"toy_a": ont}
    assert_no_reference_leakage([_row(A + "Cardiomyopathy", [A + "HeartDisease"])], set(), onts)
    with pytest.raises(AssertionError):  # negative from another ontology
        assert_no_reference_leakage([_row(A + "Cardiomyopathy", ["http://example.org/b#DCM1A"])], set(), onts)
    with pytest.raises(AssertionError):  # a reference pair inside a row
        refs = {(A + "Cardiomyopathy", A + "HeartDisease")}
        assert_no_reference_leakage([_row(A + "Cardiomyopathy", [A + "HeartDisease"])], refs, onts)


def test_save_and_load_rows(tmp_path, ont):
    rows, stats = mine_ontology(ont, pairs_cfg(hard_negatives=["sibling"]), None, seed=3)
    path = tmp_path / "p.jsonl"
    save_rows(rows, stats, [], path)
    assert load_rows(path) == rows
    assert path.with_suffix(".stats.json").exists()
