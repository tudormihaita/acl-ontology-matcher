"""Training-pair mining (SPEC §4.2). Pairs are mined per ontology, never from mappings.

Each row is ``(anchor, positive, negatives)`` for one class C of one ontology:
- positives: ``syn`` (two labels of C) and ``lex`` (a label of C and a lexicon variant of it);
- hard negatives: labels of siblings / parents / children of C, with an optional filter;
  rows without an available hard negative are filled with a random label of another class
  of the same ontology (a BERTMap-style soft negative), so every row has the same shape.

A negative is never a string that is also a label of C.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from contrastive_om.lexicon import Lexicon, build_lexicon, save_lexicons
from contrastive_om.ontology import Ontology

log = logging.getLogger(__name__)

HARD_NEGATIVE_KINDS = {"sibling", "sibling_filtered", "hierarchy"}


@dataclass
class PairRow:
    anchor: str
    positive: str
    negatives: list[str]
    class_iri: str
    negative_class_iris: list[str]
    ontology: str
    source: str  # "syn" | "lex"
    negative_kinds: list[str] = field(default_factory=list)  # per negative: kind or "random"

    def texts(self) -> list[str]:
        return [self.anchor, self.positive, *self.negatives]


# --------------------------------------------------------------------------- positives


def positive_pairs(
    ont: Ontology, lexicon: Lexicon | None, sources: list[str], max_pairs: int, rng: random.Random
) -> list[tuple[str, str, str, str]]:
    """(anchor, positive, class_iri, source) tuples, at most ``max_pairs`` per class.

    When a class has more candidates than the cap, ``syn`` and ``lex`` candidates are
    interleaved (round-robin) so both sources are represented.
    """
    out = []
    for c in ont:  # classes are sorted (ontology.py), so this is deterministic
        pools = []
        if "syn" in sources:
            pools.append([(a, b, "syn") for a, b in itertools.permutations(c.labels, 2)])
        if "lex" in sources and lexicon is not None:
            lex = []
            for v in lexicon.variants.get(c.iri, []):
                lex += [(v.label, v.variant, "lex"), (v.variant, v.label, "lex")]
            pools.append(lex)
        for pool in pools:
            rng.shuffle(pool)
        chosen = []
        for group in itertools.zip_longest(*pools):
            chosen.extend(p for p in group if p is not None)
        seen = set()
        for a, b, src in chosen:
            if (a, b) not in seen and len(seen) < max_pairs:
                seen.add((a, b))
                out.append((a, b, c.iri, src))
    return out


# --------------------------------------------------------------------------- negatives


class NegativeSampler:
    """Samples hard negatives for a class, with the SPEC §4.2.2 filters."""

    def __init__(
        self,
        ont: Ontology,
        kinds: list[str],
        filter_cos: float = 0.85,
        label_embeddings: dict[str, np.ndarray] | None = None,
    ):
        unknown = set(kinds) - HARD_NEGATIVE_KINDS
        if unknown:
            raise ValueError(f"unsupported hard negatives: {sorted(unknown)}")
        if "sibling_filtered" in kinds and label_embeddings is None:
            raise ValueError("sibling_filtered needs label embeddings from the pre-trained encoder")
        self.ont = ont
        self.kinds = kinds
        self.filter_cos = filter_cos
        self.emb = label_embeddings
        self.has_disjointness = any(c.disjoint_with for c in ont)
        self._with_labels = [c.iri for c in ont if c.labels]

    def _class_candidates(self, iri: str) -> list[tuple[str, str]]:
        """(negative class, kind) candidates for class ``iri``, deterministic order."""
        out = []
        if "sibling" in self.kinds or "sibling_filtered" in self.kinds:
            kind = "sibling_filtered" if "sibling_filtered" in self.kinds else "sibling"
            out += [(s, kind) for s in sorted(self.ont.siblings(iri))]
        if "hierarchy" in self.kinds:
            near = self.ont.informative_parents(iri) | self.ont[iri].children
            out += [(n, "hierarchy") for n in sorted(near)]
        return out

    def _allowed(self, iri: str, neg_iri: str, neg_label: str, anchor: str, kind: str) -> bool:
        own = self.ont[iri]
        if neg_iri == iri or neg_label in own.labels:
            return False
        if kind == "sibling_filtered" and not self._sibling_passes_filter(iri, neg_iri, anchor):
            return False
        return True

    def _sibling_passes_filter(self, iri: str, neg_iri: str, anchor: str) -> bool:
        """The filter drops the whole sibling *class* (not single labels), otherwise a sibling
        whose name is too close would still be used through its abbreviation."""
        own, sib = self.ont[iri], self.ont[neg_iri]
        if set(sib.labels) & set(own.labels):  # (a) shares a label with C
            return False
        if self.emb is not None and anchor in self.emb:  # (b) any label too close to the anchor
            a = self.emb[anchor]
            if max(float(a @ self.emb[lab]) for lab in sib.labels) > self.filter_cos:
                return False
        if self.has_disjointness and neg_iri not in own.disjoint_with:  # (c)
            return False
        return True

    def sample(self, iri: str, anchor: str, k: int, rng: random.Random) -> list[tuple[str, str, str]]:
        """Up to ``k`` (label, class, kind) hard negatives, from distinct classes."""
        candidates = self._class_candidates(iri)
        rng.shuffle(candidates)
        out, used = [], set()
        for neg_iri, kind in candidates:
            if len(out) >= k:
                break
            if neg_iri in used:
                continue
            labels = list(self.ont[neg_iri].labels)
            rng.shuffle(labels)
            for lab in labels:
                if self._allowed(iri, neg_iri, lab, anchor, kind):
                    out.append((lab, neg_iri, kind))
                    used.add(neg_iri)
                    break
        return out

    def random_fill(self, iri: str, exclude: set[str], k: int, rng: random.Random) -> list[tuple[str, str, str]]:
        """Soft negatives: random labels of other classes that share no label with C."""
        own = set(self.ont[iri].labels)
        out = []
        for _ in range(100 * max(k, 1)):
            if len(out) >= k:
                break
            neg_iri = self._with_labels[rng.randrange(len(self._with_labels))]
            if neg_iri == iri or neg_iri in exclude or set(self.ont[neg_iri].labels) & own:
                continue
            exclude.add(neg_iri)
            out.append((rng.choice(self.ont[neg_iri].labels), neg_iri, "random"))
        return out


def encode_labels(
    ont: Ontology, encoder_name: str, device: str, extra: set[str] | None = None, batch_size: int = 256
) -> dict[str, np.ndarray]:
    """L2-normalized pre-trained-encoder embeddings of every label and of ``extra`` texts
    (lexicon variants, which can be anchors), for the ``sibling_filtered`` cosine filter."""
    from sentence_transformers import SentenceTransformer

    labels = sorted({lab for c in ont for lab in c.labels} | (extra or set()))
    model = SentenceTransformer(encoder_name, device=device)
    vecs = model.encode(labels, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return dict(zip(labels, vecs))


# --------------------------------------------------------------------------- mining


def mine_ontology(
    ont: Ontology,
    cfg_pairs,
    lexicon: Lexicon | None,
    seed: int,
    label_embeddings: dict[str, np.ndarray] | None = None,
) -> tuple[list[PairRow], dict]:
    rng = random.Random(f"{seed}-{ont.name}")
    positives = list(cfg_pairs.positives)
    kinds = list(cfg_pairs.hard_negatives or [])
    k = int(cfg_pairs.num_hard_negatives) if kinds else 0  # no hard negatives = in-batch negatives only
    sampler = NegativeSampler(ont, kinds, float(cfg_pairs.filter_cos), label_embeddings)

    rows: list[PairRow] = []
    n_filled = 0
    for anchor, positive, iri, src in positive_pairs(ont, lexicon, positives, int(cfg_pairs.max_pairs_per_class), rng):
        negs = sampler.sample(iri, anchor, k, rng) if k else []
        if len(negs) < k:
            n_filled += 1
            negs += sampler.random_fill(iri, {n for _, n, _ in negs}, k - len(negs), rng)
        rows.append(
            PairRow(
                anchor=anchor,
                positive=positive,
                negatives=[lab for lab, _, _ in negs],
                class_iri=iri,
                negative_class_iris=[n for _, n, _ in negs],
                ontology=ont.name,
                source=src,
                negative_kinds=[kd for _, _, kd in negs],
            )
        )

    covered = {r.class_iri for r in rows}
    single = [c.iri for c in ont if len(c.labels) == 1]
    stats = {
        "classes": len(ont),
        "rows": len(rows),
        "rows_by_source": dict(Counter(r.source for r in rows)),
        "classes_covered": len(covered),
        "pct_classes_covered": round(100 * len(covered) / max(len(ont), 1), 2),
        "single_label_classes": len(single),
        "single_label_classes_covered": sum(1 for i in single if i in covered),
        "negatives_per_anchor": k,
        "negatives_by_kind": dict(Counter(kd for r in rows for kd in r.negative_kinds)),
        "rows_filled_with_random_negatives": n_filled,
    }
    return rows, stats


def assert_no_reference_leakage(rows: list[PairRow], references: set[tuple[str, str]], ontologies: dict[str, Ontology]):
    """SPEC §3.3 rule 5: every row stays inside one ontology and links no reference pair."""
    for r in rows:
        ont = ontologies[r.ontology]
        assert r.class_iri in ont.classes, f"anchor class outside its ontology: {r.class_iri}"
        for n in r.negative_class_iris:
            assert n in ont.classes, f"negative class outside the row's ontology: {n}"
            assert (r.class_iri, n) not in references and (n, r.class_iri) not in references, (
                f"reference pair in mined pairs: {r.class_iri} / {n}"
            )


def mine_pairs(
    cfg,
    ontologies: dict[str, Ontology],
    references: set[tuple[str, str]] | None = None,
    label_fixes: dict[str, list] | None = None,
) -> tuple[list[PairRow], dict, list[Lexicon]]:
    """Mine rows for every ontology (``{"source": ont, "target": ont}``) as configured."""
    positives = list(cfg.pairs.positives)
    lexicons: dict[str, Lexicon] = {}
    if "lex" in positives:
        pooled = None
        if cfg.lex.get("rule_scope", "per_ontology") == "pooled":
            pooled = [c.labels for o in ontologies.values() for c in o]
        for which, ont in ontologies.items():
            lexicons[which] = build_lexicon(ont, cfg.lex, (label_fixes or {}).get(which), rule_groups=pooled)

    embeddings = {}
    if "sibling_filtered" in (cfg.pairs.hard_negatives or []):
        from contrastive_om.config import resolve_device

        device = resolve_device(cfg.get("device", "auto"))
        for which, ont in ontologies.items():
            variants = {v.variant for vs in lexicons[which].variants.values() for v in vs} if which in lexicons else set()
            embeddings[which] = encode_labels(ont, cfg.model.encoder, device, extra=variants)

    rows: list[PairRow] = []
    stats: dict = {}
    for which, ont in ontologies.items():
        r, s = mine_ontology(ont, cfg.pairs, lexicons.get(which), int(cfg.seed), embeddings.get(which))
        rows += r
        stats[ont.name] = s
        if which in lexicons:
            stats[ont.name]["lexicon"] = lexicons[which].stats

    if references is not None:
        assert_no_reference_leakage(rows, references, {o.name: o for o in ontologies.values()})
        stats["leakage_check"] = "passed"
    return rows, stats, list(lexicons.values())


# --------------------------------------------------------------------------- I/O


def pairs_path(cfg) -> Path:
    digest = cfg.hash("task", "ontology", "pairs", "lex", "model", "seed")
    return Path(cfg.output.pairs_dir) / f"{digest}.jsonl"


def save_rows(rows: list[PairRow], stats: dict, lexicons: list[Lexicon], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
    path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    if lexicons:
        save_lexicons(lexicons, path.with_suffix(".lexicon.json"))


def load_rows(path: str | Path) -> list[PairRow]:
    with open(path) as fh:
        return [PairRow(**json.loads(line)) for line in fh]
