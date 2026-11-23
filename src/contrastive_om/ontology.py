"""OWL ontology parsing: named classes with labels, definitions and direct hierarchy.

Only *literal* values of the configured label/definition properties are read.
Cross-reference properties (``hasDbXref``, ``seeAlso``, ``closeMatch``, ...) are
never read, so no mapping information can leak into labels (SPEC §3.3, rule 4).
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import rdflib
from rdflib import OWL, RDF, RDFS, BNode, Literal, URIRef
from rdflib.collection import Collection

log = logging.getLogger(__name__)

PREFIXES = {
    "rdf": str(RDF),
    "rdfs": str(RDFS),
    "owl": str(OWL),
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
    "obo": "http://purl.obolibrary.org/obo/",
    "efo": "http://www.ebi.ac.uk/efo/",
    "biomlann": "http://oaei.ontologymatching.org/bio-ml/ann/",
}

# Properties that carry cross-references to other resources. They are never read,
# and configuring one of them as a label/definition property is an error.
# skos:exactMatch is included: in Bio-ML's OMIM it holds name strings of unclear
# provenance (possibly names of linked UMLS/MeSH concepts), see data/README.md.
XREF_PROPERTIES = frozenset(
    PREFIXES["oboInOwl"] + p for p in ("hasDbXref",)
) | frozenset(
    PREFIXES["skos"] + p for p in ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch")
) | {str(RDFS.seeAlso), str(OWL.sameAs)}

USE_IN_ALIGNMENT = URIRef(PREFIXES["biomlann"] + "use_in_alignment")

_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")  # lower->Upper only, so "CMD1A" stays intact
_WS = re.compile(r"\s+")


def expand_curie(curie: str) -> str:
    """``rdfs:label`` -> full IRI. Full IRIs are returned unchanged."""
    if curie.startswith("http://") or curie.startswith("https://"):
        return curie
    prefix, _, local = curie.partition(":")
    if prefix not in PREFIXES or not local:
        raise ValueError(f"unknown CURIE prefix in '{curie}'")
    return PREFIXES[prefix] + local


def normalise_label(text: str) -> str:
    """Lowercase, split camelCase/underscores, collapse whitespace.

    camelCase is only split for single-token identifiers such as ``ConferenceDinner``,
    so natural-language labels like ``McCune-Albright syndrome`` are left intact.
    """
    text = str(text).replace("_", " ").strip()
    if " " not in text:
        text = _CAMEL.sub(" ", text)
    return _WS.sub(" ", text.lower()).strip()


@dataclass
class OntologyClass:
    iri: str
    labels: list[str] = field(default_factory=list)  # normalised, deduplicated, rdfs:label first
    definitions: list[str] = field(default_factory=list)  # raw text
    raw_labels: list[str] = field(default_factory=list)  # un-normalised (for abbreviation mining)
    parents: set[str] = field(default_factory=set)
    children: set[str] = field(default_factory=set)
    disjoint_with: set[str] = field(default_factory=set)
    alignable: bool = True  # False if Bio-ML marks it use_in_alignment=false


class Ontology:
    """A parsed ontology.

    ``max_parent_fanout``: parents with more direct children than this are treated as
    generic categories (e.g. ORDO's "disease", 3.9k children). They are kept in
    ``parents``/``children`` but ignored by :meth:`informative_parents` and therefore by
    sibling computation, hierarchy negatives and neighbor-aware scoring.
    """

    def __init__(self, name: str, classes: dict[str, OntologyClass], max_parent_fanout: int | None = None):
        self.name = name
        self.classes = classes
        self.max_parent_fanout = max_parent_fanout
        self._ancestors: dict[str, frozenset[str]] = {}

    def __len__(self) -> int:
        return len(self.classes)

    def __getitem__(self, iri: str) -> OntologyClass:
        return self.classes[iri]

    def __iter__(self):
        return iter(self.classes.values())

    @property
    def alignable_iris(self) -> list[str]:
        """Classes that may appear in predicted mappings."""
        return [c.iri for c in self if c.alignable and c.labels]

    def informative_parents(self, iri: str) -> set[str]:
        """Direct parents that are not generic categories (see ``max_parent_fanout``)."""
        parents = self.classes[iri].parents
        if self.max_parent_fanout is None:
            return set(parents)
        return {p for p in parents if len(self.classes[p].children) <= self.max_parent_fanout}

    def siblings(self, iri: str) -> set[str]:
        """Classes sharing at least one informative direct parent with ``iri`` (excluding itself)."""
        out: set[str] = set()
        for p in self.informative_parents(iri):
            out |= self.classes[p].children
        out.discard(iri)
        return out

    def ancestors(self, iri: str) -> frozenset[str]:
        """Transitive (named) superclasses, cached."""
        if iri not in self._ancestors:
            seen: set[str] = set()
            stack = list(self.classes[iri].parents)
            while stack:
                p = stack.pop()
                if p not in seen:
                    seen.add(p)
                    stack.extend(self.classes[p].parents)
            self._ancestors[iri] = frozenset(seen)
        return self._ancestors[iri]

    @cached_property
    def depth(self) -> dict[str, int]:
        """Shortest distance from a root (a class with no named parents); roots have depth 0."""
        depth = {c.iri: 0 for c in self if not c.parents}
        queue = deque(depth)
        while queue:
            iri = queue.popleft()
            for child in self.classes[iri].children:
                if child not in depth:
                    depth[child] = depth[iri] + 1
                    queue.append(child)
        return depth

    @cached_property
    def longest_depth(self) -> dict[str, int]:
        """Longest path from a root (cycle-safe). Shows the depth of fine classifications that a
        shallow shortcut (e.g. ORDO's type taxonomy) hides from :attr:`depth`."""
        memo: dict[str, int] = {}
        for start in self.classes:
            if start in memo:
                continue
            stack, on_path = [(start, iter(self.classes[start].parents))], {start}
            best = {start: 0}
            while stack:
                node, it = stack[-1]
                parent = next(it, None)
                if parent is None:
                    stack.pop()
                    on_path.discard(node)
                    memo[node] = best[node]
                    if stack:
                        child = stack[-1][0]
                        best[child] = max(best[child], memo[node] + 1)
                elif parent in memo:
                    best[node] = max(best[node], memo[parent] + 1)
                elif parent not in on_path:
                    on_path.add(parent)
                    best[parent] = 0
                    stack.append((parent, iter(self.classes[parent].parents)))
        return memo

    @cached_property
    def label_index(self) -> dict[str, set[str]]:
        """Normalised label -> IRIs of classes carrying it."""
        index: dict[str, set[str]] = {}
        for c in self:
            for label in c.labels:
                index.setdefault(label, set()).add(c.iri)
        return index


def _is_true(graph: rdflib.Graph, subject, predicate) -> bool:
    return any(str(o).lower() == "true" for o in graph.objects(subject, predicate))


def apply_label_fixes(label: str, fixes: list[dict] | None) -> str:
    """Apply documented data-cleaning substitutions to a normalised label.

    Each fix is ``{pattern: <regex>, replace: <str>, count: <int, 0 = all>}``.
    Used for the Bio-ML OMIM artefact where "type" became "iia" (see data/README.md).
    """
    for fix in fixes or []:
        label = re.sub(fix["pattern"], fix["replace"], label, count=int(fix.get("count", 0)))
    return label


def load_ontology(
    path: str | Path,
    label_properties: list[str],
    definition_properties: list[str] | None = None,
    name: str | None = None,
    graph: rdflib.Graph | None = None,
    parent_restriction_properties: list[str] | None = None,
    max_parent_fanout: int | None = None,
    label_fixes: list[dict] | None = None,
) -> Ontology:
    """Parse an OWL (RDF/XML) file into an :class:`Ontology`.

    ``parent_restriction_properties``: superclass restrictions ``P some X`` (X named) with P in
    this list are treated as parent edges, e.g. ``obo:BFO_0000050`` (part_of) for ORDO, whose
    clinical classification is expressed that way.
    """
    path = Path(path)
    restriction_props = {URIRef(expand_curie(p)) for p in (parent_restriction_properties or [])}
    label_props = [URIRef(expand_curie(p)) for p in label_properties]
    def_props = [URIRef(expand_curie(p)) for p in (definition_properties or [])]
    for p in label_props + def_props:
        if str(p) in XREF_PROPERTIES:
            raise ValueError(f"{p} is a cross-reference property and may not be read (SPEC §3.3)")

    if graph is None:
        graph = rdflib.Graph()
        graph.parse(path, format="xml")

    deprecated = OWL.deprecated
    # Sorted so that class order (and everything mined from it) is identical across runs;
    # set iteration order of strings changes with Python's per-process hash seed.
    iris = sorted(
        {
            s
            for s in graph.subjects(RDF.type, OWL.Class)
            if isinstance(s, URIRef) and s != OWL.Thing and not _is_true(graph, s, deprecated)
        },
        key=str,
    )

    classes: dict[str, OntologyClass] = {}
    for s in iris:
        raw: list[str] = []
        for p in label_props:  # property order = label priority (rdfs:label first)
            for o in sorted(graph.objects(s, p), key=str):
                if isinstance(o, Literal) and str(o).strip():
                    raw.append(str(o).strip())
        labels = list(dict.fromkeys(apply_label_fixes(normalise_label(r), label_fixes) for r in raw))
        labels = [lab for lab in labels if lab]
        definitions = [
            str(o).strip()
            for p in def_props
            for o in graph.objects(s, p)
            if isinstance(o, Literal) and str(o).strip()
        ]
        alignable = not any(str(o).lower() == "false" for o in graph.objects(s, USE_IN_ALIGNMENT))
        classes[str(s)] = OntologyClass(
            iri=str(s),
            labels=labels,
            definitions=list(dict.fromkeys(definitions)),
            raw_labels=list(dict.fromkeys(raw)),
            alignable=alignable,
        )

    for s, o in graph.subject_objects(RDFS.subClassOf):
        if restriction_props and isinstance(o, BNode) and graph.value(o, OWL.onProperty) in restriction_props:
            o = graph.value(o, OWL.someValuesFrom)  # "P some X" -> X
            if not isinstance(o, URIRef):
                continue
        s, o = str(s), str(o)
        if s in classes and o in classes and s != o:  # named parents only
            classes[s].parents.add(o)
            classes[o].children.add(s)

    for s, o in graph.subject_objects(OWL.disjointWith):
        s, o = str(s), str(o)
        if s in classes and o in classes:
            classes[s].disjoint_with.add(o)
            classes[o].disjoint_with.add(s)
    for group in graph.subjects(RDF.type, OWL.AllDisjointClasses):
        members = [str(m) for m in Collection(graph, graph.value(group, OWL.members))]
        members = [m for m in members if m in classes]
        for m in members:
            classes[m].disjoint_with.update(x for x in members if x != m)

    ont = Ontology(name or path.stem, classes, max_parent_fanout=max_parent_fanout)
    log.info("loaded %s: %d classes (%d alignable)", ont.name, len(ont), len(ont.alignable_iris))
    return ont


def load_from_config(cfg, which: str, graph: rdflib.Graph | None = None, path: str | Path | None = None) -> Ontology:
    """Load the task's ``source`` or ``target`` ontology as configured.

    ``path`` overrides the configured file (used by the ``match`` demo on arbitrary OWL files);
    per-side label fixes then do not apply.
    """
    fixes = cfg.task.get(f"{which}_label_fixes") if path is None else None
    return load_ontology(
        path or cfg.task[f"{which}_owl"],
        label_properties=list(cfg.ontology.label_properties),
        definition_properties=list(cfg.ontology.get("definition_properties") or []),
        graph=graph,
        parent_restriction_properties=list(cfg.ontology.get("parent_restriction_properties") or []),
        max_parent_fanout=cfg.ontology.get("max_parent_fanout"),
        label_fixes=[f.to_dict() if hasattr(f, "to_dict") else f for f in (fixes or [])],
    )


# --------------------------------------------------------------------------- stats


def annotation_property_counts(path: str | Path, graph: rdflib.Graph | None = None) -> dict[str, dict]:
    """Count every predicate used on named classes, split by literal / IRI / blank-node values."""
    if graph is None:
        graph = rdflib.Graph()
        graph.parse(Path(path), format="xml")
    classes = {s for s in graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    counts: dict[str, Counter] = {}
    for s, p, o in graph:
        if s in classes:
            kind = "literal" if isinstance(o, Literal) else "iri" if isinstance(o, URIRef) else "bnode"
            counts.setdefault(str(p), Counter())[kind] += 1
    return {p: dict(c) for p, c in sorted(counts.items(), key=lambda kv: -sum(kv[1].values()))}


def xref_counts(graph: rdflib.Graph) -> dict[str, dict[str, int]]:
    """Cross-reference annotations on classes, by property and target namespace/prefix."""
    out: dict[str, Counter] = {}
    for p in sorted(XREF_PROPERTIES):
        for s, o in graph.subject_objects(URIRef(p)):
            text = str(o)
            if isinstance(o, URIRef):
                ns = re.sub(r"[^/#]+$", "", text)
            elif re.match(r"^[A-Za-z][\w.-]*:\S+$", text):
                ns = text.split(":", 1)[0] + ":"  # CURIE-like literal, e.g. "OMIM:100100"
            else:
                ns = "free-text literal"
            out.setdefault(p, Counter())[ns] += 1
    return {p: dict(c) for p, c in out.items()}


def ontology_stats(ont: Ontology) -> dict:
    n_labels = [len(c.labels) for c in ont]
    n_sib = [len(ont.siblings(c.iri)) for c in ont]
    depths = list(ont.depth.values())
    n = max(len(ont), 1)
    return {
        "classes": len(ont),
        "alignable_classes": len(ont.alignable_iris),
        "classes_without_labels": sum(1 for x in n_labels if x == 0),
        "labels_per_class_mean": round(statistics.mean(n_labels), 3) if n_labels else 0,
        "labels_per_class_median": statistics.median(n_labels) if n_labels else 0,
        "pct_ge2_labels": round(100 * sum(x >= 2 for x in n_labels) / n, 2),
        "pct_exactly1_label": round(100 * sum(x == 1 for x in n_labels) / n, 2),
        "pct_with_definition": round(100 * sum(bool(c.definitions) for c in ont) / n, 2),
        "roots": sum(1 for c in ont if not c.parents),
        "depth_max": max(depths) if depths else 0,
        "depth_mean": round(statistics.mean(depths), 3) if depths else 0,
        "longest_depth_max": max(ont.longest_depth.values(), default=0),
        "longest_depth_mean": round(statistics.mean(ont.longest_depth.values()), 3) if len(ont) else 0,
        "unreachable_from_root": len(ont) - len(depths),  # only possible with cycles
        "pct_with_parent": round(100 * sum(bool(c.parents) for c in ont) / n, 2),
        "siblings_per_class_mean": round(statistics.mean(n_sib), 3) if n_sib else 0,
        "siblings_per_class_median": statistics.median(n_sib) if n_sib else 0,
        "pct_with_sibling": round(100 * sum(x > 0 for x in n_sib) / n, 2),
        "disjointness_pairs": sum(len(c.disjoint_with) for c in ont) // 2,
    }
