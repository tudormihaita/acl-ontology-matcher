import pytest

from contrastive_om.config import Config
from contrastive_om.lexicon import (
    Rule,
    apply_rule,
    build_lexicon,
    comma_inversion,
    detokenize,
    edit_rule,
    extract_abbreviations,
    mine_rules,
    table_variants,
    tokenize,
)
from contrastive_om.ontology import Ontology, OntologyClass


def test_tokenize_roundtrip():
    for label in ["cardiomyopathy, dilated, 1a", "mitochondrial complex 5 (atp synthase) deficiency", "x"]:
        assert detokenize(tokenize(label)) == label
    assert tokenize("cardiomyopathy, dilated, 1a") == ["cardiomyopathy", ",", "dilated", ",", "1a"]


# --------------------------------------------------------------------------- (a) abbreviations


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Prune belly syndrome (PBS) is a rare disorder.", [("Prune belly syndrome", "PBS")]),
        ("A form of glycogen storage disease type 7 (GSD7).", [("glycogen storage disease type 7", "GSD7")]),
        (
            "Mitochondrial encephalomyopathy, lactic acidosis, and stroke-like episodes (MELAS)",
            [("Mitochondrial encephalomyopathy, lactic acidosis, and stroke-like episodes", "MELAS")],
        ),
        ("A heart condition (see below).", []),  # lowercase: not a short form
        ("Some disorder (OMIM).", []),  # characters cannot be matched
    ],
)
def test_extract_abbreviations(text, expected):
    assert extract_abbreviations(text) == expected


# --------------------------------------------------------------------------- (b) rules


def test_edit_rule():
    assert edit_rule("x syndrome", "x") == Rule("del", "syndrome")
    assert edit_rule("familial a", "hereditary a") == Rule("sub", "familial", "hereditary")
    assert edit_rule("a b", "c d") is None  # two edits
    assert edit_rule("a, b", "a b") is None  # punctuation only


def test_mine_rules_threshold_and_numerals():
    groups = [["x syndrome", "x"]] * 3 + [["familial a", "hereditary a"]] * 3 + [["type 2", "type ii"]] * 5
    kept, counts, _ = mine_rules(groups, min_count=3)
    assert {r.key() for r in kept} == {"del:syndrome", "sub:familial<->hereditary"}  # numerals left to table
    assert counts[Rule("sub", "2", "ii")] == 5
    assert mine_rules(groups, min_count=4)[0] == []


def test_apply_rule():
    sub = Rule("sub", "familial", "hereditary")
    assert apply_rule(sub, "familial vocal cord dysfunction") == ["hereditary vocal cord dysfunction"]
    assert apply_rule(sub, "hereditary vocal cord dysfunction") == ["familial vocal cord dysfunction"]
    assert apply_rule(Rule("del", "syndrome"), "sotos syndrome 2") == ["sotos 2"]
    assert apply_rule(Rule("del", "syndrome"), "syndrome") == []  # never produce an empty label
    assert apply_rule(sub, "cardiomyopathy") == []


# --------------------------------------------------------------------------- (c) table


def test_table_numerals_respect_context():
    assert "ehlers-danlos syndrome type 2" in table_variants("ehlers-danlos syndrome type ii")
    assert "glycogenosis type ii" in table_variants("glycogenosis type 2")
    assert "factor 8 deficiency" in table_variants("factor viii deficiency")
    assert table_variants("fragile x syndrome") == []  # "x" is not a numeral here
    assert table_variants("hepatitis b 2") == []  # "2" without context is left alone


def test_table_spelling_both_directions():
    variants = table_variants("hemolytic anemia")
    assert "haemolytic anemia" in variants and "hemolytic anaemia" in variants
    assert "tumor of the esophagus" in table_variants("tumour of the esophagus")
    assert table_variants("hemiplegia") == []  # "hemi" is not "hemo"


@pytest.mark.parametrize(
    "label, expected",
    [
        ("cardiomyopathy, dilated, 1a", "dilated cardiomyopathy 1a"),
        ("deafness, autosomal recessive 17", "autosomal recessive deafness 17"),
        ("immunodeficiency, common variable, 11", "common variable immunodeficiency 11"),
        ("renal dysplasia, bilateral", "bilateral renal dysplasia"),
        ("microcephalic primordial dwarfism, montreal type", None),  # named subtype
        ("palmar, plantar and disseminated porokeratosis", None),  # list, not an inversion
        ("46,xy disorder of sex development", None),  # karyotype comma
        ("citrullinemia, type ii", None),  # only an index after the comma
        ("autoimmune disease, susceptibility to, 2", None),
    ],
)
def test_comma_inversion(label, expected):
    assert comma_inversion(label) == expected


def test_comma_inversion_is_opt_in():
    assert "dilated cardiomyopathy 1a" not in table_variants("cardiomyopathy, dilated, 1a")
    assert "dilated cardiomyopathy 1a" in table_variants("cardiomyopathy, dilated, 1a", comma_inversion_on=True)


# --------------------------------------------------------------------------- build_lexicon


def _toy() -> Ontology:
    spec = {
        "C1": (["X syndrome", "X"], []),
        "C2": (["Y syndrome", "Y"], []),
        "C3": (["Z syndrome"], []),  # single label: gets "z" from the mined rule
        "C4": (["W"], []),
        "C5": (["W syndrome"], []),  # rule would give "w", a label of C4: dropped
        "C6": (["Prune belly syndrome"], ["Prune belly syndrome (PBS) is rare."]),
        "C7": (["Mitochondrial disease (MITO)"], []),
    }
    classes = {}
    for iri, (raw, defs) in spec.items():
        labels = [r.lower() for r in raw]
        classes[iri] = OntologyClass(iri=iri, labels=labels, raw_labels=raw, definitions=defs)
    return Ontology("toy", classes)


LEX = Config({"abbreviations": True, "rewrite_rules": True, "normalization_table": True, "min_rule_count": 2})


def test_build_lexicon_variants_and_filters():
    lex = build_lexicon(_toy(), LEX)
    variants = {iri: {v.variant: v.mechanism for v in vs} for iri, vs in lex.variants.items()}
    assert [r["rule"] for r in lex.rules] == ["del:syndrome"]
    assert variants["C3"] == {"z": "rule:del:syndrome"}
    assert "C5" not in variants  # "w" belongs to C4
    assert lex.stats["dropped"]["label_of_other_class"] == 1
    assert variants["C6"]["pbs"] == "abbrev"  # from the definition, long form = C6's label
    assert variants["C6"]["prune belly"] == "rule:del:syndrome"
    assert variants["C7"] == {"mitochondrial disease": "abbrev", "mito": "abbrev"}
    assert "C1" not in variants  # "x" is already a label of C1
    assert lex.stats["single_label_classes_with_variants"] == 3  # C3, C6, C7


def test_build_lexicon_switches():
    lex = build_lexicon(_toy(), Config({**LEX.to_dict(), "rewrite_rules": False, "abbreviations": False}))
    assert lex.rules == [] and lex.variants == {}


def test_pooled_rule_groups():
    # rules mined from external groups are applied to this ontology's labels
    lex = build_lexicon(_toy(), Config({**LEX.to_dict(), "abbreviations": False}),
                        rule_groups=[["familial z syndrome", "hereditary z syndrome"]] * 2)
    assert [r["rule"] for r in lex.rules] == ["sub:familial<->hereditary"]
    assert lex.variants == {}  # no toy label contains "familial"/"hereditary"
