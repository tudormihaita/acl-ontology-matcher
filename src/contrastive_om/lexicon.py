"""Internal lexicon (SPEC §4.2.1): label variants mined from an ontology's own content.

Three mechanisms produce *variants* of a class's labels, which become extra positive pairs:

(a) **Abbreviations** (Schwartz & Hearst, 2003) found in raw labels and definitions,
    e.g. "Mitochondrial ... episodes (MELAS)" -> ("mitochondrial ... episodes", "melas").
(b) **Rewrite rules** mined from pairs of labels of the *same class*: single-token
    substitutions ("familial" <-> "hereditary") and single-token deletions ("x syndrome" -> "x"),
    kept if they occur at least ``min_rule_count`` times.
(c) A small hand-written **normalization table**: roman <-> arabic numerals and
    British <-> American spelling.

No mapping between ontologies is ever used. Rules are mined from within-class synonymy only;
variants are always variants of the class's own labels.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from contrastive_om.ontology import Ontology, apply_label_fixes, normalize_label

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- tokens

_TOKEN = re.compile(r"[^\s,;()]+|[,;()]")


def tokenize(label: str) -> list[str]:
    """Split a normalized label into word tokens and punctuation tokens (, ; ( ))."""
    return _TOKEN.findall(label)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,;)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r",\s*,", ",", text)
    return re.sub(r"\s+", " ", text).strip(" ,;")


def _is_word(token: str) -> bool:
    return token not in {",", ";", "(", ")"}


# --------------------------------------------------------------------------- (a) abbreviations


def _valid_short_form(sf: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", sf)
    return (
        2 <= len(sf) <= 10
        and len(sf.split()) <= 2
        and sf[0].isalnum()
        and bool(letters)
        and any(ch.isupper() for ch in sf)  # avoids "(see below)", "(rare)", ...
    )


def _best_long_form(sf: str, candidate: str) -> str | None:
    """Schwartz & Hearst: match short-form characters right-to-left inside the candidate text;
    the first short-form character must start a word."""
    s_idx, l_idx = len(sf) - 1, len(candidate) - 1
    while s_idx >= 0:
        ch = sf[s_idx].lower()
        if not ch.isalnum():
            s_idx -= 1
            continue
        while l_idx >= 0 and (
            candidate[l_idx].lower() != ch or (s_idx == 0 and l_idx > 0 and candidate[l_idx - 1].isalnum())
        ):
            l_idx -= 1
        if l_idx < 0:
            return None
        l_idx -= 1
        s_idx -= 1
    start = candidate.rfind(" ", 0, l_idx + 1) + 1
    long_form = candidate[start:].strip()
    if len(long_form) <= len(sf) or sf.lower() in long_form.lower().split():
        return None
    return long_form


def extract_abbreviations(text: str) -> list[tuple[str, str]]:
    """(long form, short form) pairs from "long form (SF)" patterns in raw text."""
    pairs = []
    for m in re.finditer(r"\(([^()]+)\)", text):
        sf = re.split(r"[;,]", m.group(1))[0].strip()
        if not _valid_short_form(sf):
            continue
        before = re.split(r"[.;:!?]\s", text[: m.start()])[-1].strip()  # same sentence/clause
        words = before.split()
        n_letters = len(re.sub(r"[^A-Za-z0-9]", "", sf))
        window = " ".join(words[-min(n_letters + 5, 2 * n_letters):])
        lf = _best_long_form(sf, window)
        if lf:
            pairs.append((lf, sf))
    return pairs


# --------------------------------------------------------------------------- (b) rewrite rules


@dataclass(frozen=True)
class Rule:
    kind: str  # "sub" (a <-> b) or "del" (drop token)
    a: str
    b: str = ""

    def key(self) -> str:
        return f"{self.kind}:{self.a}<->{self.b}" if self.kind == "sub" else f"del:{self.a}"


def edit_rule(label_a: str, label_b: str) -> Rule | None:
    """The single-token edit turning one label into the other, if there is exactly one."""
    ta, tb = tokenize(label_a), tokenize(label_b)
    ops = [op for op in SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes() if op[0] != "equal"]
    if len(ops) != 1:
        return None
    tag, i1, i2, j1, j2 = ops[0]
    if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1 and _is_word(ta[i1]) and _is_word(tb[j1]):
        a, b = sorted((ta[i1], tb[j1]))
        return Rule("sub", a, b)
    if tag in ("delete", "insert"):
        span = ta[i1:i2] if tag == "delete" else tb[j1:j2]
        if len(span) == 1 and _is_word(span[0]):
            return Rule("del", span[0])
    return None


def mine_rules(label_groups: list[list[str]], min_count: int) -> tuple[list[Rule], Counter, dict[str, list]]:
    """Count single-token edits over all label pairs of each group (= one class)."""
    counts: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    for labels in label_groups:
        for x, y in itertools.combinations(labels, 2):
            rule = edit_rule(x, y)
            if rule is not None:
                counts[rule] += 1
                if len(examples[rule.key()]) < 3:
                    examples[rule.key()].append([x, y])
    kept = sorted(
        (r for r, n in counts.items() if n >= min_count and not _is_numeral_rule(r)),
        key=lambda r: (-counts[r], r.key()),
    )
    return kept, counts, examples


def _is_numeral_rule(rule: Rule) -> bool:
    """Numeral substitutions ("2" <-> "ii") are left to the context-aware normalization table."""
    numerals = set(_ROMAN) | set(_ARABIC)
    return rule.kind == "sub" and rule.a in numerals and rule.b in numerals


def apply_rule(rule: Rule, label: str) -> list[str]:
    """Variants of ``label`` under ``rule`` (first matching occurrence; both directions for subs)."""
    tokens = tokenize(label)
    out = []
    if rule.kind == "sub":
        for src, dst in ((rule.a, rule.b), (rule.b, rule.a)):
            if src in tokens:
                i = tokens.index(src)
                out.append(detokenize(tokens[:i] + [dst] + tokens[i + 1:]))
    elif rule.kind == "del" and rule.a in tokens and sum(map(_is_word, tokens)) > 1:
        i = tokens.index(rule.a)
        out.append(detokenize(tokens[:i] + tokens[i + 1:]))
    return out


# --------------------------------------------------------------------------- (c) normalization table

_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}
_ARABIC = {v: k for k, v in _ROMAN.items()}
# Words after which a number is a type/subtype index. "i", "v" and "x" are only converted in
# this context ("fragile x syndrome" stays as is); the other numerals are unambiguous.
_NUMBER_CONTEXT = {"type", "types", "subtype", "class", "complex", "group", "stage", "grade", "form", "factor", "phase"}
_AMBIGUOUS_ROMAN = {"i", "v", "x"}
# American <-> British spelling, matched as token prefixes in both directions. Only prefixes that
# are unambiguous are listed ("hemo", not "hem", so "hemiplegia" is never touched).
_SPELLING = {
    "hemo": "haemo", "hemat": "haemat", "hemorrh": "haemorrh", "anemi": "anaemi", "leukemi": "leukaemi",
    "ischemi": "ischaemi", "glycemi": "glycaemi", "lipemi": "lipaemi", "esophag": "oesophag",
    "edema": "oedema", "pediatr": "paediatr", "orthoped": "orthopaed", "fetal": "foetal",
    "tumor": "tumour", "celiac": "coeliac", "diarrhea": "diarrhoea", "dyspnea": "dyspnoea",
}
_SPELLING_BOTH = {**_SPELLING, **{uk: us for us, uk in _SPELLING.items()}}


_INDEX_TOKEN = re.compile(r"^(\d+[a-z]?|[a-z]?\d+|[ivx]+[a-z]?)$")


_NOT_A_MODIFIER = {"and", "or", "with", "without", "to", "of", "due", "in", "for", "by", "type", "form", "types",
                   "susceptibility", "associated", "related"}


def comma_inversion(label: str) -> str | None:
    """Inverted heading -> natural order: "cardiomyopathy, dilated, 1a" -> "dilated cardiomyopathy 1a".

    The first segment is the head; the following segments are adjective-like modifiers placed
    before it, in reverse order; a trailing index ("1a", "type 2", or a number ending the last
    segment) stays at the end. Conservative: modifiers must be 1-3 words without conjunctions,
    prepositions or "type"/"form" (so "dwarfism, montreal type" is left alone).
    """
    segments = [s.strip() for s in label.split(", ")]
    if not 2 <= len(segments) <= 3 or not all(segments):
        return None
    index: list[str] = []
    last = segments[-1].split()
    if len(last) == 2 and last[0] in _NUMBER_CONTEXT and _INDEX_TOKEN.match(last[1]):
        index = last
        segments.pop()
    elif _INDEX_TOKEN.match(last[-1]):
        index = [last[-1]]
        if len(last) == 1:
            segments.pop()
        else:
            segments[-1] = " ".join(last[:-1])
    if len(segments) < 2:
        return None
    head, modifiers = segments[0], segments[1:]
    for m in modifiers:
        words = m.split()
        if not 1 <= len(words) <= 3 or set(words) & _NOT_A_MODIFIER or any(ch.isdigit() for ch in m):
            return None
    return " ".join(list(reversed(modifiers)) + [head] + index)


def table_variants(label: str, comma_inversion_on: bool = False) -> list[str]:
    tokens = tokenize(label)
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i else ""
        if tok in _ROMAN and (tok not in _AMBIGUOUS_ROMAN or prev in _NUMBER_CONTEXT):
            out.append(detokenize(tokens[:i] + [_ROMAN[tok]] + tokens[i + 1:]))
        elif tok in _ARABIC and prev in _NUMBER_CONTEXT:
            out.append(detokenize(tokens[:i] + [_ARABIC[tok]] + tokens[i + 1:]))
        # longest matching prefix only (so "haemato..." uses "haemat", not "haem...")
        matches = [p for p in _SPELLING_BOTH if tok.startswith(p)]
        if matches:
            src = max(matches, key=len)
            out.append(detokenize(tokens[:i] + [_SPELLING_BOTH[src] + tok[len(src):]] + tokens[i + 1:]))
    if comma_inversion_on:
        inverted = comma_inversion(label)
        if inverted:
            out.append(inverted)
    return out


# --------------------------------------------------------------------------- lexicon


@dataclass
class Variant:
    label: str  # the class label it was derived from
    variant: str
    mechanism: str  # "abbrev" | "rule:<key>" | "table"


@dataclass
class Lexicon:
    ontology: str
    rules: list[dict] = field(default_factory=list)  # kept rules with counts and examples
    abbreviations: list[list[str]] = field(default_factory=list)  # [long, short, iri]
    variants: dict[str, list[Variant]] = field(default_factory=dict)  # iri -> variants
    stats: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "ontology": self.ontology,
            "stats": self.stats,
            "rules": self.rules,
            "abbreviations": self.abbreviations[:500],
            "variant_examples": {
                iri: [asdict(v) for v in vs[:5]] for iri, vs in itertools.islice(self.variants.items(), 50)
            },
        }


def build_lexicon(
    ont: Ontology,
    cfg_lex,
    label_fixes: list[dict] | None = None,
    rule_groups: list[list[str]] | None = None,
) -> Lexicon:
    """Build the lexicon of one ontology.

    ``rule_groups``: label groups to mine rules from. ``None`` mines from this ontology's own
    classes (``rule_scope: per_ontology``); the caller passes both ontologies' groups for
    ``rule_scope: pooled``. Variants are always generated from this ontology's own labels.
    """
    get = cfg_lex.get if hasattr(cfg_lex, "get") else dict(cfg_lex).get
    lex = Lexicon(ont.name)
    candidates: dict[str, list[Variant]] = defaultdict(list)

    def norm(text: str) -> str:
        return apply_label_fixes(normalize_label(text), label_fixes)

    # (a) abbreviations: from a class's own raw labels, or from definitions when the long form
    # is exactly a label of some class of this ontology.
    n_abbrev_found = 0
    if get("abbreviations", True):
        for c in ont:
            sources = [(r, True) for r in c.raw_labels] + [(d, False) for d in c.definitions]
            for text, from_label in sources:
                for lf, sf in extract_abbreviations(text):
                    n_abbrev_found += 1
                    lf_n, sf_n = norm(lf), norm(sf)
                    if from_label:
                        targets = [c.iri]
                        stripped = norm(re.sub(r"\s*\([^()]*\)", "", text))
                        if stripped and stripped != norm(text):
                            candidates[c.iri].append(Variant(norm(text), stripped, "abbrev"))
                    else:
                        targets = sorted(ont.label_index.get(lf_n, ()))
                    for iri in targets:
                        lex.abbreviations.append([lf_n, sf_n, iri])
                        candidates[iri].append(Variant(lf_n if lf_n in ont[iri].labels else ont[iri].labels[0],
                                                       sf_n, "abbrev"))

    # (b) rewrite rules
    rules: list[Rule] = []
    if get("rewrite_rules", True):
        groups = rule_groups if rule_groups is not None else [c.labels for c in ont]
        rules, counts, examples = mine_rules(groups, int(get("min_rule_count", 20)))
        lex.rules = [{"rule": r.key(), "count": counts[r], "examples": examples[r.key()]} for r in rules]
        for c in ont:
            for label in c.labels:
                for rule in rules:
                    for v in apply_rule(rule, label):
                        candidates[c.iri].append(Variant(label, v, f"rule:{rule.key()}"))

    # (c) normalization table
    if get("normalization_table", True):
        invert = bool(get("comma_inversion", False))
        for c in ont:
            for label in c.labels:
                for v in table_variants(label, comma_inversion_on=invert):
                    candidates[c.iri].append(Variant(label, v, "table"))

    # filters: must be new for the class, must not be another class's label, deduplicated
    dropped = Counter()
    for iri in sorted(candidates):
        own = set(ont[iri].labels)
        seen: set[str] = set()
        kept = []
        for v in candidates[iri]:
            if not v.variant or v.variant in own:
                dropped["already_a_label"] += 1
            elif ont.label_index.get(v.variant, set()) - {iri}:
                dropped["label_of_other_class"] += 1
            elif v.variant in seen:
                dropped["duplicate"] += 1
            else:
                seen.add(v.variant)
                kept.append(v)
        if kept:
            lex.variants[iri] = kept

    by_mech = Counter(v.mechanism.split(":")[0] for vs in lex.variants.values() for v in vs)
    single = [c.iri for c in ont if len(c.labels) == 1]
    lex.stats = {
        "abbreviations_found": n_abbrev_found,
        "rules_kept": len(rules),
        "variants_total": sum(by_mech.values()),
        "variants_by_mechanism": dict(by_mech),
        "classes_with_variants": len(lex.variants),
        "single_label_classes": len(single),
        "single_label_classes_with_variants": sum(1 for i in single if i in lex.variants),
        "dropped": dict(dropped),
    }
    log.info("lexicon %s: %s", ont.name, lex.stats)
    return lex


def save_lexicons(lexicons: list[Lexicon], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([lx.to_json() for lx in lexicons], indent=2))
