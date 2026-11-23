"""The batch rules of SPEC §4.2.3 must hold for every batch."""

import random

import pytest

from contrastive_om.batching import batch_violations, build_batches
from contrastive_om.pairs import PairRow


def row(cls, anchor, positive, negs=(), ontology="A"):
    negs = list(negs)
    return PairRow(anchor, positive, [f"label of {n}" for n in negs], cls, negs, ontology, "syn",
                   ["sibling"] * len(negs))


def random_rows(seed=0, n_classes=40, rows_per_class=4):
    """Two ontologies; several rows per class; random same-ontology negatives; shared strings."""
    rng = random.Random(seed)
    rows = []
    for ont in ("A", "B"):
        classes = [f"{ont}{i}" for i in range(n_classes)]
        for c in classes:
            for k in range(rows_per_class):
                neg = rng.choice([x for x in classes if x != c])
                pos = "shared name" if (c.endswith("7") and k == 0) else f"{c} syn {k}"  # string on 4 classes
                rows.append(row(c, f"{c} label", pos, [neg], ont))
    return rows


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [4, 16, 64])
def test_all_rules_hold_and_every_row_used_once(seed, batch_size):
    rows = random_rows(seed)
    batches = build_batches(rows, batch_size, seed=seed)
    assert sorted(i for b in batches for i in b) == list(range(len(rows)))
    for b in batches:
        assert len(b) <= batch_size
        assert batch_violations(rows, b) == []


def test_per_ontology_batches():
    rows = random_rows()
    for b in build_batches(rows, 32, seed=0):
        assert len({rows[i].ontology for i in b}) == 1


def test_class_unique():
    rows = [row("C", f"c{k}", f"c{k}'") for k in range(5)]  # one class, five rows
    batches = build_batches(rows, 8, seed=0)
    assert len(batches) == 5 and all(len(b) == 1 for b in batches)


def test_no_hard_negative_from_in_batch_anchor_class():
    # C's row has a negative from D; D's row from E; E's row from C: no two may share a batch
    rows = [row("C", "c", "c'", ["D"]), row("D", "d", "d'", ["E"]), row("E", "e", "e'", ["C"])]
    batches = build_batches(rows, 3, seed=0)
    assert len(batches) == 3


def test_shared_positive_text_is_split():
    rows = [row("C1", "a", "shared"), row("C2", "b", "shared"), row("C3", "c", "c'")]
    for b in build_batches(rows, 3, seed=0):
        texts = [t for i in b for t in (rows[i].anchor, rows[i].positive)]
        assert len(texts) == len(set(texts))


def test_negative_equal_to_another_rows_anchor_is_split():
    # C2's negative text is C1's anchor text (a string shared by two classes)
    rows = [row("C1", "x", "x'"), PairRow("y", "y'", ["x"], "C2", ["C9"], "A", "syn", ["sibling"])]
    assert len(build_batches(rows, 2, seed=0)) == 2


def test_repeated_negatives_may_share_a_batch():
    # siblings with the same parent as hard negative: allowed together
    rows = [PairRow(f"s{i}", f"s{i}'", ["parent"], f"S{i}", ["P"], "A", "syn", ["hierarchy"]) for i in range(4)]
    assert build_batches(rows, 4, seed=0) == [[0, 1, 2, 3]] or len(build_batches(rows, 4, seed=0)) == 1


def test_deterministic_and_seed_dependent():
    rows = random_rows()
    assert build_batches(rows, 16, seed=5) == build_batches(rows, 16, seed=5)
    assert build_batches(rows, 16, seed=5) != build_batches(rows, 16, seed=6)


def test_batches_are_mostly_full():
    # 600 rows per ontology = 18 full batches of 32 + 24 rows that cannot fill a batch.
    # Partial batches may hold only (about) that unavoidable remainder.
    rows = random_rows(n_classes=200, rows_per_class=3)
    batches = build_batches(rows, 32, seed=0)
    rows_in_partial = sum(len(b) for b in batches if len(b) < 32)
    assert rows_in_partial <= 2 * 32


def test_batch_violations_detects_each_rule():
    rows = [
        row("C", "a", "b", ["D"], "A"),
        row("C", "c", "d", [], "A"),  # same class
        row("D", "e", "f", [], "A"),  # D is a negative of row 0
        row("E", "g", "b", [], "A"),  # text "b" repeated
        row("F", "h", "i", [], "B"),  # other ontology
    ]
    problems = " | ".join(batch_violations(rows, [0, 1, 2, 3, 4]))
    for expected in ("duplicate class_iri", "hard negative from in-batch class", "mixed ontologies",
                     "duplicate anchor/positive text"):
        assert expected in problems
    assert batch_violations(rows, [0]) == []
    clash = [row("C", "a", "b"), PairRow("c", "d", ["a"], "E", ["Z"], "A", "syn", ["sibling"])]
    assert batch_violations(clash, [0, 1]) == ["negative equals an anchor/positive text"]


def test_unknown_scope():
    with pytest.raises(ValueError):
        build_batches([], 8, seed=0, scope="global")
