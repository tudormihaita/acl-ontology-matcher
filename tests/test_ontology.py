from pathlib import Path

import pytest

from contrastive_om.ontology import (
    apply_label_fixes,
    load_ontology,
    normalize_label,
    ontology_stats,
)

FIX = Path(__file__).parent / "fixtures"
A = "http://example.org/a#"
B = "http://example.org/b#"
LABELS = ["rdfs:label", "skos:prefLabel", "skos:altLabel", "oboInOwl:hasExactSynonym", "efo:alternative_term"]
DEFS = ["obo:IAO_0000115", "skos:definition", "efo:definition"]


@pytest.fixture(scope="module")
def ont_a():
    return load_ontology(FIX / "toy_a.owl", LABELS, DEFS)


@pytest.fixture(scope="module")
def ont_b():
    return load_ontology(FIX / "toy_b.owl", LABELS, DEFS)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("HeartDisease", "heart disease"),
        ("ConferenceDinner", "conference dinner"),
        ("cardiac   disease", "cardiac disease"),
        ("McCune-Albright syndrome", "mccune-albright syndrome"),  # multi-word: no camel split
        ("early_registered_author", "early registered author"),
        ("CMD1A", "cmd1a"),  # abbreviations stay one token
        ("NF2", "nf2"),
        ("  Dilated Cardiomyopathy 1A ", "dilated cardiomyopathy 1a"),
    ],
)
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_labels_deduplicated_with_rdfs_label_first(ont_a):
    assert ont_a[A + "DilatedCardiomyopathy1A"].labels == [
        "dilated cardiomyopathy 1a",  # rdfs:label first
        "cmd1a",  # then synonyms, sorted by raw string ("CMD1A" < "Cardiomyopathy, ...")
        "cardiomyopathy, dilated, 1a",  # duplicate "dilated cardiomyopathy 1a" removed
    ]


def test_related_synonyms_off_by_default(ont_a):
    assert "heart failure" not in ont_a[A + "DilatedCardiomyopathy1A"].labels


def test_alternative_term_is_a_label(ont_b):
    assert ont_b[B + "CardiacMuscleDisease"].labels == ["cardiac muscle disease", "cardiomyopathy"]


def test_deprecated_and_thing_skipped(ont_a):
    assert A + "ObsoleteThing" not in ont_a.classes
    assert "http://www.w3.org/2002/07/owl#Thing" not in ont_a.classes
    assert len(ont_a) == 7


def test_hierarchy_named_parents_only(ont_a):
    cmp_ = ont_a[A + "Cardiomyopathy"]
    assert cmp_.parents == {A + "HeartDisease"}  # restriction ignored
    assert cmp_.children == {A + "DilatedCardiomyopathy1A", A + "DilatedCardiomyopathy1B"}
    assert ont_a[A + "Disease"].parents == set()  # owl:Thing dropped


def test_siblings_and_ancestors(ont_a):
    assert ont_a.siblings(A + "DilatedCardiomyopathy1A") == {A + "DilatedCardiomyopathy1B"}
    assert ont_a.siblings(A + "HeartDisease") == {A + "LungDisease", A + "HelperClass"}
    assert ont_a.ancestors(A + "DilatedCardiomyopathy1A") == {
        A + "Cardiomyopathy",
        A + "HeartDisease",
        A + "Disease",
    }


def test_depth(ont_a):
    assert ont_a.depth[A + "Disease"] == 0
    assert ont_a.depth[A + "DilatedCardiomyopathy1A"] == 3


def test_longest_depth_sees_past_shortcuts():
    from contrastive_om.ontology import Ontology, OntologyClass

    # root -> a -> b -> c, plus a shortcut root -> c
    cls = {k: OntologyClass(iri=k, labels=[k]) for k in ("root", "a", "b", "c")}
    for child, parent in (("a", "root"), ("b", "a"), ("c", "b"), ("c", "root")):
        cls[child].parents.add(parent)
        cls[parent].children.add(child)
    ont = Ontology("t", cls)
    assert ont.depth["c"] == 1
    assert ont.longest_depth["c"] == 3
    assert ont.longest_depth["root"] == 0


def test_disjointness(ont_a, ont_b):
    assert ont_a[A + "HeartDisease"].disjoint_with == {A + "LungDisease"}
    assert ont_a[A + "LungDisease"].disjoint_with == {A + "HeartDisease"}
    assert ont_b[B + "DCM1A"].disjoint_with == {B + "DCM1B"}  # owl:AllDisjointClasses


def test_use_in_alignment_flag(ont_a):
    assert ont_a[A + "HelperClass"].alignable is False
    assert A + "HelperClass" not in ont_a.alignable_iris
    assert A + "Cardiomyopathy" in ont_a.alignable_iris


def test_definitions(ont_a, ont_b):
    assert ont_a[A + "Cardiomyopathy"].definitions == ["A disease of the heart muscle (CMP)."]
    assert ont_b[B + "CardiacMuscleDisease"].definitions == ["Disease of the myocardium."]


def test_part_of_restrictions_ignored_by_default(ont_b):
    assert ont_b[B + "DCM1D"].parents == set()


def test_part_of_restrictions_as_parents():
    ont = load_ontology(FIX / "toy_b.owl", LABELS, parent_restriction_properties=["obo:BFO_0000050"])
    assert ont[B + "DCM1D"].parents == {B + "CardiacMuscleDisease"}
    assert B + "DCM1D" in ont.siblings(B + "DCM1A")


def test_max_parent_fanout_ignores_generic_parents():
    ont = load_ontology(
        FIX / "toy_b.owl", LABELS, parent_restriction_properties=["obo:BFO_0000050"], max_parent_fanout=3
    )
    # CardiacMuscleDisease has 4 children (1A-1D) > 3: treated as a generic category
    assert ont[B + "DCM1A"].parents == {B + "CardiacMuscleDisease"}  # still stored
    assert ont.informative_parents(B + "DCM1A") == set()
    assert ont.siblings(B + "DCM1A") == set()
    # DisorderOfHeart has 1 child: still informative
    assert ont.informative_parents(B + "CardiacMuscleDisease") == {B + "DisorderOfHeart"}


@pytest.mark.parametrize(
    "label, expected",
    [
        ("mitochondrial complex 4 deficiency, nuclear iia 7", "mitochondrial complex 4 deficiency, nuclear type 7"),
        ("myoclonic epilepsy, hartung iia", "myoclonic epilepsy, hartung type"),
        ("congenital disorder of glycosylation, iia iia", "congenital disorder of glycosylation, type iia"),
        ("glycogen storage disease", "glycogen storage disease"),
        ("siiab syndrome", "siiab syndrome"),  # word boundary
    ],
)
def test_iia_label_fix(label, expected):
    fixes = [{"pattern": r"\biia\b", "replace": "type", "count": 1}]
    assert apply_label_fixes(label, fixes) == expected


def test_label_fix_applied_at_load_time():
    fixes = [{"pattern": r"\b1a\b", "replace": "one-a", "count": 1}]
    ont = load_ontology(FIX / "toy_a.owl", LABELS, label_fixes=fixes)
    assert ont[A + "DilatedCardiomyopathy1A"].labels[0] == "dilated cardiomyopathy one-a"


# ---------------------------------------------------------------- leakage (SPEC §3.3, rule 4)


def test_xrefs_never_appear_in_labels(ont_a):
    all_text = {t for c in ont_a for t in c.labels + c.raw_labels + c.definitions}
    assert not any("999999" in t for t in all_text)  # oboInOwl:hasDbXref value
    assert not any("example.org/b" in t for t in all_text)  # rdfs:seeAlso target


@pytest.mark.parametrize(
    "prop", ["oboInOwl:hasDbXref", "rdfs:seeAlso", "skos:closeMatch", "skos:exactMatch"]
)
def test_xref_property_cannot_be_configured(prop):
    with pytest.raises(ValueError, match="cross-reference"):
        load_ontology(FIX / "toy_a.owl", ["rdfs:label", prop])


def test_only_literal_values_are_read(ont_b):
    # DisorderOfHeart has an skos:altLabel whose value is an IRI of the other ontology:
    # label properties only contribute literal values.
    assert ont_b[B + "DisorderOfHeart"].labels == ["disorder of heart"]


def test_class_order_is_deterministic(ont_a):
    assert list(ont_a.classes) == sorted(ont_a.classes)


# ---------------------------------------------------------------- stats


def test_stats(ont_a):
    s = ontology_stats(ont_a)
    assert s["classes"] == 7
    assert s["alignable_classes"] == 6
    assert s["depth_max"] == 3
    assert s["disjointness_pairs"] == 1
    assert s["pct_exactly1_label"] == round(100 * 3 / 7, 2)  # Disease, LungDisease, HelperClass
