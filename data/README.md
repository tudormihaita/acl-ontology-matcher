# Data

Fetched by `bash scripts/download_data.sh` from Bio-ML (OAEI 2024), Zenodo record
[13119437](https://zenodo.org/records/13119437), file `omim-ordo.zip` (4.5 MB).
Everything under `data/raw/`, `data/splits/` and `data/pairs/` is git-ignored.

## Layout (as downloaded)

```
data/raw/bio-ml/omim-ordo/
  omim.owl                   11 MB   source ontology, 9,648 classes (RDF/XML)
  ordo.owl                   53 MB   target ontology, 9,275 classes (RDF/XML)
  refs_equiv/
    full.tsv                 3,721 equivalence mappings  (SrcEntity, TgtEntity, Score=1.0)
    train.tsv                1,116 mappings (30%)  -- semi-supervised training split
    test.tsv                 2,605 mappings (70%)
    test.cands.tsv           2,605 rows (SrcEntity, TgtEntity, TgtCandidates)
  refs_subs/                 subsumption task (not used)
    train.tsv  test.cands.tsv
```

Verified facts:
- `train.tsv` and `test.tsv` are disjoint and their union is exactly `full.tsv`.
- **There is no unsupervised `val` split** in this release; the provided split is 30/70.
- `test.cands.tsv` has exactly one row per `test.tsv` mapping. `TgtCandidates` is a Python
  tuple literal of **101** IRIs: the gold target plus 100 negatives.
- Mappings are almost 1-to-1: 3,699 distinct sources and 3,712 distinct targets.
- Source IRIs: `http://omim.org/entry/*` (3,367) and `http://www.omim.org/phenotypicSeries/PS*` (354).

## Ontology content (from `cli stats`, see `results/data_stats.json`)

| | OMIM | ORDO |
|---|---|---|
| named classes | 9,648 | 9,275 |
| `use_in_alignment=false` (never in refs; excluded from matching) | 6 | 437 |
| label property | `rdfs:label` | `rdfs:label` |
| synonym property | `oboInOwl:hasExactSynonym` (6,440) | `efo:alternative_term` (12,308) |
| definitions | none | `efo:definition` (71% of classes) |
| named `subClassOf` | entry → phenotypic series only (depth ≤ 1) | Orphanet *type* taxonomy only ("disease", "group of disorders", ...) |
| other hierarchy | — | `part_of some X` restrictions (14,878 edges): the clinical classification |
| other restrictions | — | `has_inheritance`, `has_age_of_onset` (attributes) |
| disjointness axioms | 0 | 0 |
| `owl:deprecated` | 0 | 0 |
| `oboInOwl:hasDbXref` | 0 | 0 |

Other predicates, ignored: OMIM `IAO_0000142` (PubMed links), `biolink:category`,
`RO_0002525` (chromosome location), `RO_0003303`; ORDO `skos:notation` (`ORPHA:NNN` codes),
`efo:definition_citation`.

## Decisions (taken after the Phase 0 report)

1. **Validation split.** No unsupervised val split is provided.
   → `refs_equiv/train.tsv` (1,116) is the **validation** set: it is never used for training.
   `test.tsv` and `test.cands.tsv` are the test set, unchanged.
2. **OMIM `skos:exactMatch`** holds 19,139 *literal strings* (not IRIs), e.g.
   "eagle-barrett syndrome", "aneurysm, abdominal aortic". 10,010 are not among the class's
   own labels/synonyms (4,713 classes). Provenance is unclear (OMIM alternative titles, or
   names of linked UMLS/MeSH concepts). → **Not read** (conservative, SPEC §3.3).
3. **ORDO hierarchy.** Using only named `subClassOf` makes ORDO flat (depth 2, about 2,100
   siblings per class). → Parents also include `part_of some X`; parents with more than 200
   children are ignored for siblings/negatives/neighbors. Result: ORDO longest-path depth 12
   (mean 7), 98% of classes with siblings (median 12).
4. **"iia" artefact in OMIM.** The token "type" was replaced by "iia" in 956 OMIM labels
   (e.g. "mitochondrial complex 4 deficiency, nuclear iia 7"; original OMIM: "... NUCLEAR TYPE 7").
   ORDO uses "type" (2,337 occurrences). → The first "iia" in each OMIM label is restored to
   "type" (`task.source_label_fixes`), for every method including the baseline.
