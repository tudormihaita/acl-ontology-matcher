"""Batch construction for InfoNCE (SPEC §4.2.3).

In MNRL every anchor is scored against *all* positives and hard negatives of its batch, so a
batch must not contain anything that is secretly a positive. Rules enforced here:

1. **Class-unique:** no two rows with the same ``class_iri``.
2. **No hard negative from an in-batch anchor class:** no row's negative belongs to a class
   that is the ``class_iri`` of a row in the batch.
3. **Per-ontology:** all rows come from one ontology (``batch_scope: per_ontology``), so a class
   and its true cross-ontology match never meet as a false negative.
4. **No text conflicts:** anchors and positives are all distinct, and no negative equals any
   anchor or positive of the batch. A string shared by two classes therefore never shows up
   as a "negative" for itself. Negatives may repeat among themselves (a duplicated wrong
   candidate is harmless), which keeps batches full when siblings share a parent negative.

Batches are built greedily over a seeded shuffle; rows that conflict with the current batch are
deferred to the next one. Every row is used exactly once.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict, deque
from typing import Sequence

from contrastive_om.pairs import PairRow

log = logging.getLogger(__name__)


class _BatchState:
    def __init__(self):
        self.rows: list[int] = []
        self.classes: set[str] = set()
        self.negative_classes: set[str] = set()
        self.pos_texts: set[str] = set()  # anchors and positives
        self.neg_texts: set[str] = set()

    @staticmethod
    def self_consistent(row: PairRow) -> bool:
        return row.anchor != row.positive and not {row.anchor, row.positive} & set(row.negatives) \
            and row.class_iri not in row.negative_class_iris

    def accepts(self, row: PairRow) -> bool:
        own = {row.anchor, row.positive}
        return (
            self.self_consistent(row)
            and row.class_iri not in self.classes  # rule 1
            and row.class_iri not in self.negative_classes  # rule 2 (row's class is a negative here)
            and not set(row.negative_class_iris) & self.classes  # rule 2 (row's negatives are anchors here)
            and not own & (self.pos_texts | self.neg_texts)  # rule 4
            and not set(row.negatives) & self.pos_texts
        )

    def add(self, idx: int, row: PairRow) -> None:
        self.rows.append(idx)
        self.classes.add(row.class_iri)
        self.negative_classes.update(row.negative_class_iris)
        self.pos_texts.update((row.anchor, row.positive))
        self.neg_texts.update(row.negatives)


def build_batches(rows: Sequence[PairRow], batch_size: int, seed: int, scope: str = "per_ontology") -> list[list[int]]:
    """Row-index batches satisfying all rules; batch order is shuffled across ontologies."""
    if scope not in ("per_ontology", "mixed"):
        raise ValueError(f"unknown batch_scope '{scope}'")
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.ontology if scope == "per_ontology" else "all"].append(i)

    batches: list[list[int]] = []
    for key in sorted(groups):
        order = groups[key][:]
        rng.shuffle(order)
        pending = deque(order)
        while pending:
            state, deferred = _BatchState(), []
            while pending and len(state.rows) < batch_size:
                i = pending.popleft()
                if state.accepts(rows[i]):
                    state.add(i, rows[i])
                else:
                    deferred.append(i)
            if not state.rows:  # even an empty batch rejects them: rows that conflict with themselves
                log.warning("dropping %d self-conflicting rows", len(deferred))
                break
            pending.extendleft(reversed(deferred))  # retry deferred rows first, in order
            batches.append(state.rows)
    rng.shuffle(batches)
    return batches


def batch_stats(batches: list[list[int]], batch_size: int) -> dict:
    sizes = [len(b) for b in batches]
    return {
        "batches": len(batches),
        "rows": sum(sizes),
        "full_batches": sum(s == batch_size for s in sizes),
        "min_size": min(sizes, default=0),
        "batches_below_half": sum(s < batch_size / 2 for s in sizes),
        "size_histogram": dict(sorted(Counter(min(s, batch_size) // 16 * 16 for s in sizes).items())),
    }


def batch_violations(rows: Sequence[PairRow], batch: Sequence[int], scope: str = "per_ontology") -> list[str]:
    """All rule violations in one batch (empty list = valid). Used by tests and as a runtime check."""
    problems = []
    chosen = [rows[i] for i in batch]
    classes = [r.class_iri for r in chosen]
    if len(set(classes)) != len(classes):
        problems.append("duplicate class_iri")
    class_set = set(classes)
    for r in chosen:
        if set(r.negative_class_iris) & class_set:
            problems.append(f"hard negative from in-batch class ({r.class_iri})")
    if scope == "per_ontology" and len({r.ontology for r in chosen}) > 1:
        problems.append("mixed ontologies")
    pos_texts = [t for r in chosen for t in (r.anchor, r.positive)]
    if len(set(pos_texts)) != len(pos_texts):
        problems.append("duplicate anchor/positive text")
    if set(pos_texts) & {n for r in chosen for n in r.negatives}:
        problems.append("negative equals an anchor/positive text")
    return problems
