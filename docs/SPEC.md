# SPEC — Contrastive Ontology Matching (ACL course, RR2)

Implementation spec for a Claude Code session. Work **phase by phase** (Section 9). Do not start a phase until the previous phase's acceptance checks pass.

Deadline: **3 days**. The core (Phases 0–7) must be finished first. Phase 8 (presentation) follows. Extensions (Section 10) only if time remains.

---

## 1. Goal

**Research question.** Can a bi-encoder trained with a **contrastive (InfoNCE) objective** on positive/negative pairs mined from each ontology produce a reusable **entity embedding space** in which ontology matching works by **nearest-neighbour search alone**, without candidate selection, rule-based mapping extension or logical repair?

**Context (from RR1).**
- DeepAlignment (Kolyvakis et al., 2018) refines static word vectors with synonym attract / antonym repel and matches directly in that space (set-to-set cosine distance, then Stable Marriage).
- BERTMap (He et al., 2022) mines synonym/non-synonym label pairs from the ontologies and fine-tunes a cross-encoder classifier. Because it is a cross-encoder, it needs a sub-word candidate index and adds rule-based extension and logic-based repair. Its results show that richer label sets are the strongest lever (FMA–SNOMED F1 0.49 → 0.83 with extra synonyms).
- This project keeps BERTMap-style self-mined pairs, replaces the classifier with a contrastive bi-encoder, and matches in the embedding space the way DeepAlignment does.
- To enrich positives without external data, it adds an **internal lexicon** (Section 4.2.1): naming variants mined from each ontology's own synonyms and applied to the rest of that ontology.

**Hard constraints.**
- **Unsupervised setting:** no reference mappings are ever used for training.
- **Test references are never used for any decision.** That includes thresholds, hyperparameters, early stopping and the choice of the "best" setting in each experiment. All such choices use the **validation** split.
- **No rule-based or logic-based post-processing in the core.** That kind of processing is only allowed in Extensions. (Numeric scoring options such as CSLS and neighbour-aware scoring, Section 4.5, are part of the core.)

---

## 2. Environment

- **Hardware:** Mac mini M4, PyTorch **MPS** backend (`device="mps"`). Fallback: Google Colab T4 (`device="cuda"`). The device is set in the config, with an `auto` mode: cuda > mps > cpu.
- **MPS notes:**
  - Set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
  - **No fp16/bf16 mixed precision on MPS**: train in fp32.
  - Keep `max_seq_length` small (labels are short; default 32).
- **Python:** 3.12, managed with **uv** (`.venv`, `uv.lock`). Install with `uv sync --extra dev`; run commands with `uv run ...`.
- **Naming:** the distribution is `acl-ontology-matcher` (pyproject); the import package is `contrastive_om`.
- **Core dependencies:** `torch`, `sentence-transformers>=3`, `datasets`, `rdflib`, `numpy`, `pandas`, `scipy`, `pyyaml`, `tqdm`, `pytest`, `umap-learn`, `matplotlib`, `jupyter`.
- **Not needed for the core:** Java / DeepOnto. They may be used only in Extension 10.2.

---

## 3. Data

### 3.1 Core task: Bio-ML OMIM–ORDO (equivalence, unsupervised setting)

- **Source:** Bio-ML, OAEI 2024 version, Zenodo record **13119437** (https://zenodo.org/records/13119437). Docs: https://krr-oxford.github.io/DeepOnto/bio-ml.
- **Task:** OMIM ↔ ORDO, disease domain, from MONDO. About 9.6k source classes, about 9.3k target classes, 3,721 reference equivalence mappings.
- **Download:** `scripts/download_data.sh` fetches the Bio-ML archive and extracts only the OMIM–ORDO task into `data/raw/bio-ml/omim-ordo/`. If the download fails (network or file naming), stop and ask the user to place the files manually.
- **Inspect the layout:** in Phase 0, print the actual folder layout and document it in `data/README.md`. Expected contents, to be verified:
  - source and target `.owl` files;
  - reference mappings as TSV (`SrcEntity`, `TgtEntity`, `Score`), with full/test and possibly train/val splits;
  - `test.cands.tsv` for local ranking (`SrcEntity`, `TgtEntity`, `TgtCandidates`, with 100 candidates per row).

### 3.2 Splits

- **Decided after Phase 0.** The release ships only a 30/70 split (`refs_equiv/train.tsv` 1,116 / `refs_equiv/test.tsv` 2,605) and no unsupervised validation split (`data/README.md`).
  - **Validation = Bio-ML `train.tsv`** (1,116 mappings). It is never used for training, only for θ and for choosing settings.
  - **Test = Bio-ML `test.tsv`** (2,605), with the official `test.cands.tsv` (101 candidates per row, gold included) unchanged, since it aligns exactly with `test.tsv`.
- **Null references:** when scoring on test, predictions involving validation mappings are ignored, and vice versa.
- **Training never uses mappings.** Pairs are mined from all classes of both ontologies, including those that appear in validation/test mappings. This is transductive and standard in unsupervised OM: the model sees each ontology's text and structure, never the correspondences.

### 3.3 Leakage rules

1. **Training pairs come only from the internal content of each ontology:** labels, synonyms, definitions and hierarchy. Never from mappings.
2. **No encoder pre-trained on UMLS synonym pairs** (e.g. SapBERT), because the Bio-ML ontologies overlap with UMLS and MONDO.
3. **No external lexicons or auxiliary ontologies in the core.** UMLS, MONDO, DOID, MeSH and Wikidata link OMIM and Orphanet directly or indirectly, so their synonym pairs can encode gold mappings. LLM-generated synonyms are excluded for the same reason (memorisation).
4. **Cross-references are never read.** `oboInOwl:hasDbXref`, `skos:exactMatch`/`closeMatch`, `rdfs:seeAlso` and `owl:equivalentClass` to external IRIs are excluded from labels, definitions and pairs. Only literal values of the configured label properties are read. A unit test asserts that no xref value appears in the parsed labels.
   - Phase 0 finding: the Bio-ML files contain no `hasDbXref`.
   - OMIM's `skos:exactMatch` holds 19k **literal name strings** of unclear provenance: possibly OMIM alternative titles, possibly names of linked UMLS/MeSH concepts. Decision: **not read**.
5. **The pair miner asserts that no (source-class, target-class) reference pair appears in its output.** This is trivially true when mining per ontology, but keep the assert.

### 3.4 Optional second dataset (Extension 10.4): OAEI Conference

- **What it is:** 16 small conference-organisation ontologies (OntoFarm). The 21 reference alignments cover the 7 that have gold mappings. Each ontology has about 14–140 classes and usually one label per class (camelCase local names).
- **Source:** https://oaei.ontologymatching.org/2025/conference/. Use `rar2` as the main reference.
- **Why include it:** it is label-poor, so synonym positives barely exist, and it is DeepAlignment's benchmark. It tests the structural-view positives (`struct`, Section 10.4).

---

## 4. Method

### 4.1 Ontology parsing (`src/contrastive_om/ontology.py`)

Parsing uses `rdflib`. For each named class, collect:
- `iri`
- `labels`: a deduplicated list of normalised strings. Lowercase, split camelCase and underscores, collapse whitespace.
- `definitions`: from configurable definition properties (default `obo:IAO_0000115`, `skos:definition`), if present. Used only as input to abbreviation extraction (Section 4.2.1).
- `parents` and `children`: direct, named classes only. Drop `owl:Thing` and anonymous restrictions.
- `siblings`: classes sharing at least one direct parent.
- `ancestors`: transitive, cached.
- `disjoint_with`, if present.

**Label annotation properties** are configurable. Default set: `rdfs:label`, `skos:prefLabel`, `skos:altLabel`, `oboInOwl:hasExactSynonym`, `efo:alternative_term` (ORDO's synonym property). Optional: `oboInOwl:hasRelatedSynonym`, `hasNarrowSynonym`, `hasBroadSynonym`, which are noisy and off by default. Cross-reference properties are never read (Section 3.3, rule 4).

**Hierarchy** (decided after Phase 0):
- Parent edges are named `rdfs:subClassOf` **plus** `P some X` restrictions for P in `parent_restriction_properties` (default `obo:BFO_0000050`, part_of). ORDO's named `subClassOf` is only Orphanet's type taxonomy ("disease", "group of disorders", ...), and its clinical classification is expressed with part_of.
- Parents with more than `max_parent_fanout` (default 200) direct children are generic categories. They are kept as parents but ignored ("non-informative") for siblings, hierarchy negatives and neighbour-aware scoring.
- With these settings, 98% of ORDO classes have siblings (median 12), and OMIM is unaffected (its largest phenotypic series has 102 members).

**Label fixes** (per side, documented data cleaning, applied to every method including the baseline):
- In Bio-ML's OMIM, "type" was replaced by "iia" in about 960 labels (e.g. "…, nuclear iia 7").
- `task.source_label_fixes` restores the first "iia" in each label to "type". "…, iia iia" was originally "TYPE IIA" and becomes "type iia".

In Phase 0, print which annotation properties exist and how often, including definition and cross-reference properties.

**Stats to report per ontology:**
- number of classes;
- labels per class (mean, median, % with ≥2 labels, % with exactly 1 label);
- % of classes with a definition;
- number of cross-reference annotations, by target namespace;
- depth;
- siblings per class;
- number of disjointness axioms.

Save to `results/data_stats.json`.

Skip obsolete or deprecated classes (`owl:deprecated true`).

### 4.2 Pair mining (`src/contrastive_om/pairs.py`)

Mine **per ontology**, independently. Output rows look like `{anchor, positive, negatives:[...], class_iri, negative_class_iris:[...], ontology, source}` and are saved to `data/pairs/<config-hash>.jsonl`. Also save a stats file with pair counts per positive source, the % of classes covered, the % of single-label classes that received at least one positive, and the negatives per anchor.

**Positive sources** (config `positives`, list):

| key | anchor | positive | notes |
|---|---|---|---|
| `syn` | label *a* of class C | another label *b* of C | all ordered pairs, capped at `max_pairs_per_class` (default 10) |
| `lex` | a label of C | a variant of a label of C produced by the internal lexicon (Section 4.2.1) | gives positives to single-label classes; capped at `max_pairs_per_class` together with `syn` |

#### 4.2.1 Internal lexicon (`src/contrastive_om/lexicon.py`)

The lexicon turns each ontology's own naming conventions into new positive pairs. It is built **per ontology** from that ontology's content only. Three mechanisms:

**(a) Abbreviation extraction.** Run the Schwartz–Hearst algorithm over the raw (un-normalised) labels and definitions to find (long form, short form) pairs. Example: `"alzheimer disease 2; AD2"` → `("alzheimer disease 2", "ad2")`. A pair is kept only if the long form is a label of C or contained in a label/definition of C. It becomes a positive for C.

**(b) Rewrite rules mined from synonym pairs.**
- For every pair of labels of the same class, compute the token-level edit after normalisation. Rule types:
  - comma inversion: `"x, y"` ↔ `"y x"` (e.g. `"deafness, autosomal dominant 5"` ↔ `"autosomal dominant deafness 5"`);
  - single-token insertion/deletion (e.g. `"type 2"` ↔ `"2"`);
  - single-token substitution (e.g. `"syndrome"` ↔ `"disease"`).
- Represent each edit as a context-free rule and count its occurrences. Keep rules with count ≥ `lex.min_rule_count` (default 20).
- Apply the kept rules to every label of every class to generate variants.

**(c) Normalisation table** (hand-written, small, in `lexicon.py`):
- roman ↔ arabic numerals I–X ↔ 1–10 when they appear as standalone tokens (e.g. `"type ii"` ↔ `"type 2"`);
- a short British/US spelling list (e.g. `haem`/`hem`, `oesophag`/`esophag`, `tumour`/`tumor`).

**Filters for generated variants:**
- the variant must differ from every existing label of C;
- drop the variant if it equals a normalised label of **another** class in the same ontology (it would create a contradictory positive);
- deduplicate variants; respect `max_pairs_per_class`.

**Output:** `data/pairs/<config-hash>.lexicon.json` with the kept rules, their counts and examples, the abbreviation pairs, and the number of positives each mechanism added. These examples are reused in the analysis and the presentation.

#### 4.2.2 Hard negatives

**Hard-negative sources** (config `hard_negatives`, list; in-batch negatives always apply):

| key | negative | notes |
|---|---|---|
| `sibling` | a label of a sibling class | the hardest; can be a false negative |
| `sibling_filtered` | as `sibling`, after filtering | drop a sibling if (a) it shares any normalised label with C, (b) its pre-trained-encoder cosine to the anchor is > `filter_cos` (default 0.85), or (c) the ontology has disjointness axioms and they do not declare the sibling disjoint from C. Rule (c) only applies if the ontology has any disjointness axioms; Phase 0 reports whether it does |
| `hierarchy` | a label of a parent or child of C | related but not equivalent |
| `model_mined` | (should) the nearest same-ontology labels of other classes under a model from a first training round | excludes classes sharing a normalised label with C; round 2 retrains from the base encoder with these negatives |

`num_hard_negatives` (default 1) is the number of hard negatives per row. **Rows with no available hard negative are filled with a random label from another class of the same ontology** (a BERTMap-style soft negative). Every row then has the same shape. The miner counts these filled rows.

#### 4.2.3 Batch composition

These rules are critical for InfoNCE correctness:
- **Class-unique batches.** A batch never contains two rows with the same `class_iri`. Otherwise one class's synonyms become negatives for each other.
- **No hard negative from an in-batch anchor class.** No row's hard negative may belong to a class that is the `class_iri` of any row in the same batch. Otherwise (e.g. row *i* = class C, row *j* = sibling D with a hard negative from C) C's own synonym becomes a negative for C's anchor.
- **Per-ontology batches** (`batch_scope: per_ontology`, the default). All rows in a batch come from one ontology. Otherwise a class and its true cross-ontology match can meet in a batch as a false negative.
- **Implementation:** either pre-build batches and disable trainer shuffling, or override the trainer's batch sampler. A unit test must verify all three rules.

### 4.3 Contrastive objective and training (`src/contrastive_om/train.py`)

- **Model:** `SentenceTransformer`. Default encoder `sentence-transformers/all-MiniLM-L6-v2`: 6 layers, hidden size 384, about 22M parameters, mean pooling, already contrastively pre-trained on sentence pairs.
- **Optional encoder (E6):** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (PubMedBERT): BERT-base, 110M parameters, biomedical vocabulary, no sentence-level pretraining. Mean pooling is added automatically.
- **Similarity in the loss:** cosine similarity of L2-normalised embeddings, scaled by 1/τ, followed by softmax cross-entropy. Matching uses the same cosine.
- **Loss:** InfoNCE via `losses.MultipleNegativesRankingLoss(model, scale=1/τ)`. For anchor *a_i*, positive *p_i* and hard negatives *n_ik*, the negatives are all other positives in the batch plus all hard negatives in the batch.
  - Config `loss: mnrl | mnrl_symmetric | cached_mnrl | supcon`.
  - Use `cached_mnrl` (`CachedMultipleNegativesRankingLoss`) for batch sizes above what fits in memory.
  - `supcon` (should): a custom supervised-contrastive loss in which all labels of the same class in a batch are positives for each other. This needs class-grouped batches (k labels per class), still per-ontology, and still with no hard negative from an in-batch class.
- **Defaults:** τ = 0.05 (scale 20), batch size 128, 1 epoch, lr 2e-5, warmup 10%, AdamW, weight decay 0.01, `max_seq_length` 32, seed 42.
- **Training data:** pairs from **both** ontologies of the task. This is unsupervised: no mappings are used.
- **Per-run output:** `runs/<YYYYmmdd-HHMM>-<config-name>/`, containing the config copy, `model/`, `train_log.jsonl` (loss per step), `metrics.json` and `timing.json`.

### 4.4 Class representation (`src/contrastive_om/represent.py`)

Config `class_repr`:
- `max` (default): embed every label. The score of a class pair is the maximum cosine over all their label pairs.
- `mean`: the class embedding is the L2-normalised mean of its label embeddings.

Embeddings are L2-normalised, so cosine is the dot product. Only labels are embedded; IRIs (opaque IDs) and cross-references are never used.

### 4.5 Matching from the embedding space (`src/contrastive_om/match.py`)

1. Encode all classes of the source and the target ontology once, with batched encoding cached to `.npy` in the run dir.
2. Compute source × target scores in chunks. Never materialise the full label × label matrix at once; for `max`, reduce label scores to class scores chunk by chunk.
3. **Scoring** (config `scoring`):
   - `cosine`: the class score from Section 4.4.
   - `csls`: cross-domain similarity local scaling (Conneau et al., 2018), which corrects hubness: `csls(s,t) = 2·cos(s,t) − r_T(s) − r_S(t)`, where `r_T(s)` is the mean cosine of s to its `csls_k` (default 10) nearest targets, and `r_S(t)` likewise for t over sources.
4. Keep the top-k targets per source (k = 10) and their scores.
5. **Neighbour-aware re-scoring** (config `neighbour_lambda`, default 0 = off), applied to the top-k candidates only:
   `score'(s,t) = (1−λ)·score(s,t) + λ·sim_N(s,t)`,
   where `sim_N(s,t)` = the mean over parents p of s of the max over parents q of t of `score(p,q)`. If s or t has no parents, `sim_N(s,t) = score(s,t)`, so there is no penalty. λ ∈ {0, 0.1, 0.2, 0.3} is chosen on validation.
6. **Selection** (config `selection`):
   - `argmax`: the best target per source.
   - `mutual`: keep (s, t) only if t is the best target for s **and** s is the best source for t.
   - `stable_marriage`: one-to-one Gale–Shapley over the top-k lists.
7. **Threshold:** keep a mapping if score ≥ θ. θ is chosen on the **validation** split (Section 5.2).
8. **Output:** `mappings.tsv` (`SrcEntity`, `TgtEntity`, `Score`).

No candidate index, no extension, no repair.

### 4.6 String-match baseline (`match.py`, `method: string_match`)

Map (s, t) with score 1.0 if s and t share at least one normalised label (Section 4.1 normalisation, without lexicon variants). Every such pair is kept; no threshold and no selection. It is evaluated with the same harness:
- global matching P/R/F1;
- ranking metrics, using token-Jaccard over the best label pair as the ranking score.

It is the floor the learned method must beat, and it shows where each method wins (precision vs recall on non-identical strings).

---

## 5. Evaluation (`src/contrastive_om/evaluate.py`)

### 5.1 Metrics

| Group | Metric | Computed on |
|---|---|---|
| Global matching | Precision, Recall, F1 | test refs; validation refs as null references (ignored) |
| Local ranking | MRR, Hits@1, Hits@5 | test candidate lists (each gold target ranked among its 100 candidates) |
| Full retrieval | Recall@1/5/10, MRR | each test ref's gold target ranked among **all** target classes; also computed on validation refs for model selection |
| Embedding space | alignment, uniformity (Wang & Isola, 2020) | alignment = E‖f(x)−f(y)‖² over **test reference pairs** (`mean` class embeddings; measurement only, never used for decisions); uniformity = log E exp(−2‖f(x)−f(y)‖²) over 5k random labels from both ontologies |
| Cost | training time, encoding time, matching time, peak memory if available | per run |

### 5.2 Protocol

- **Threshold:** sweep θ on **validation**, pick the θ with the best validation F1, then report test P/R/F1 at that θ. Also save the full validation P/R/F1-vs-θ curve.
  - For `cosine`: θ from 0.30 to 1.00 in steps of 0.01.
  - For `csls`: 101 evenly spaced values between the 1st and 99th percentile of the selected pairs' validation scores.
  - Log a warning if the chosen θ lies on the edge of the grid.
- **Choosing the best setting in each experiment:** highest validation F1; ties broken by validation full-retrieval MRR. Test metrics are reported but never used to choose.
- **Reference points:**
  - the string-match baseline (Section 4.6);
  - the pre-trained encoder with no fine-tuning ("epoch 0"), evaluated with the same pipeline. This shows the effect of contrastive training.
- **Robustness:** the final configuration is trained with 3 seeds (42, 43, 44) and reported as mean ± std. A bootstrap 95% CI (1,000 resamples of test mappings) is computed for its F1 and MRR.
- **Results table:** all runs append one row to `results/summary.csv`, with config name, key config fields, all metrics, θ and times.

### 5.3 Harness validation

`tests/test_evaluate.py` contains hand-made predictions and references with known P/R/F1, MRR and Hits@k, including the null-reference behaviour. All must match exactly.

---

## 6. Experiments

Each experiment is one YAML in `configs/`, run with `uv run python -m contrastive_om.cli run --config configs/<name>.yaml`. The best setting from each experiment (chosen on validation, Section 5.2) becomes the default for the next.

| ID | Varies | Values | Training runs | Priority |
|---|---|---|---|---|
| B | string-match baseline | exact normalised label overlap | 0 | must |
| E0 | epoch-0 reference | pre-trained MiniLM | 0 | must |
| E1 | positives | `[syn]` / `[syn, lex]` | 2 | must |
| E2 | hard negatives | none / `sibling` / `sibling_filtered` / `hierarchy` | 3 (none = E1 winner) | must |
| E2b | model-mined hard negatives | two-round training with `model_mined` | 1 | should |
| E3a | temperature | τ ∈ {0.02, 0.05, 0.1} | 2 (0.05 = E2 winner) | must |
| E3b | batch size | {32, 128, 512} (`cached_mnrl` for 512) | 2 | should |
| E3c | loss | `mnrl` / `supcon` | 1 | should |
| E4 | class representation | `max` / `mean` | 0 (inference only) | must |
| E5 | scoring × selection | {`cosine`, `csls`} × {`argmax`, `mutual`, `stable_marriage`} | 0 (inference only) | must |
| E5b | neighbour-aware re-scoring | λ ∈ {0, 0.1, 0.2, 0.3} | 0 (inference only) | must |
| E6 | encoder | PubMedBERT on the final config | 1 | could |
| S | seeds | final config × 3 seeds | 2 | must |

The total is about 9 "must" training runs plus up to 5 "should/could" runs. MiniLM runs should take minutes on the M4. If a single run exceeds about 20 minutes, reduce `max_pairs_per_class` or switch that run to Colab.

---

## 7. Analysis (Phase 6)

1. **Automatic error categories.**
   - For each test false positive (s, t_pred) with gold target t_gold, classify how t_pred relates to t_gold in the target ontology: `sibling`, `parent`, `child`, `ancestor/descendant (≤3 hops)`, `unrelated`. Also note whether the labels have high token overlap (Jaccard ≥ 0.5).
   - For each false negative, record the rank of t_gold and whether the source or target has only one label.
   - Report category counts for E0 vs the final model, for E1 `[syn]` vs `[syn, lex]`, and for E2 `none` vs `sibling_filtered`.
2. **Qualitative examples:** 10 FP and 10 FN examples with labels and scores, in `results/error_examples.md`.
3. **Lexicon examples:** the top mined rules with counts and example variants, and gold pairs that E1 `[syn, lex]` fixes relative to `[syn]`, in `results/lexicon_examples.md`.
4. **Embedding figure:** a UMAP of the class embeddings of one disease subtree, plus its gold-matched classes from the other ontology. Show it for E0 and for the final model, with lines connecting gold pairs. Save as `results/figures/umap_*.png`.
5. **Plots:** the validation P/R/F1-vs-θ curve, ablation bar charts for E1–E5b (with the baseline and E0 as reference lines), and a τ sweep plot. Save to `results/figures/`.

---

## 8. Repository layout

```
acl-ontology-matcher/
  CLAUDE.md  README.md  pyproject.toml  uv.lock  .gitignore
  docs/               # SPEC.md, RR1 slides, course requirements
  configs/            # base.yaml + one yaml per experiment row; smoke.yaml
  data/README.md      # actual dataset layout (Phase 0)
  data/raw/ data/splits/ data/pairs/      # git-ignored except README
  scripts/download_data.sh
  src/contrastive_om/
    __init__.py config.py ontology.py lexicon.py pairs.py batching.py
    train.py represent.py match.py evaluate.py analysis.py cli.py
  tests/fixtures/toy_a.owl toy_b.owl toy_refs.tsv
  tests/test_ontology.py test_lexicon.py test_pairs.py test_batching.py test_evaluate.py test_match.py
  notebooks/results.ipynb
  presentation/       # Phase 8 (format decided then)
  runs/               # git-ignored
  results/summary.csv results/figures/ results/*.md   # committed
```

### Config schema (`configs/base.yaml`)

```yaml
name: base
seed: 42
device: auto              # auto | mps | cuda | cpu
task:
  source_owl: data/raw/bio-ml/omim-ordo/<src>.owl
  target_owl: data/raw/bio-ml/omim-ordo/<tgt>.owl
  val_refs: data/splits/val.tsv
  test_refs: data/splits/test.tsv
  test_cands: data/splits/test.cands.tsv
ontology:
  label_properties: [rdfs:label, skos:prefLabel, skos:altLabel, oboInOwl:hasExactSynonym]
  definition_properties: [obo:IAO_0000115, skos:definition]
pairs:
  positives: [syn]          # syn | lex
  hard_negatives: []        # sibling | sibling_filtered | hierarchy | model_mined
  num_hard_negatives: 1
  max_pairs_per_class: 10
  filter_cos: 0.85
  batch_scope: per_ontology
lex:
  abbreviations: true
  rewrite_rules: true
  normalisation_table: true
  min_rule_count: 20
model:
  encoder: sentence-transformers/all-MiniLM-L6-v2
  max_seq_length: 32
train:
  enabled: true             # false => epoch-0 reference
  loss: mnrl                # mnrl | mnrl_symmetric | cached_mnrl | supcon
  temperature: 0.05
  batch_size: 128
  epochs: 1
  lr: 2.0e-5
  warmup_ratio: 0.1
match:
  method: embedding         # embedding | string_match
  class_repr: max           # max | mean
  scoring: cosine           # cosine | csls
  csls_k: 10
  neighbour_lambda: 0.0
  selection: argmax         # argmax | mutual | stable_marriage
  top_k: 10
  threshold_grid: [0.30, 1.00, 0.01]   # cosine; csls uses the percentile grid (Section 5.2)
```

`configs/smoke.yaml` subsamples about 500 classes per ontology and about 2k pairs, for end-to-end runs under 2 minutes.

### CLI (`uv run python -m contrastive_om.cli <cmd>`)

- `stats --config C`: parse the ontologies and write data stats (`results/data_stats.json`).
- `mine --config C`: mine pairs (and the lexicon, if enabled) and print pair stats.
- `train --config C`: train and save the model to a run dir.
- `match --config C --model PATH [--source X.owl --target Y.owl --threshold T --out mappings.tsv]`: works on **any** two OWL files (this is the demo).
- `evaluate --config C --run RUN_DIR`: compute all metrics and append to `summary.csv`.
- `run --config C`: mine → train → match → evaluate, end to end (or only match → evaluate for `string_match` and `train.enabled: false`).
- `inspect --model PATH --source X.owl --target Y.owl --query "label"`: print the top-k target classes for a label (demo aid).

---

## 9. Phases

Each phase ends with its acceptance checks passing. The user commits each phase (`phase N: <summary>`). The time budget assumes 3 days.

### Day 1

**Phase 0: setup and data (≈1.5h)**
- Fill in `pyproject.toml`, install dependencies with `uv sync`, and confirm `torch.backends.mps.is_available()`.
- Write `scripts/download_data.sh`, download and extract OMIM–ORDO, and write `data/README.md` with the real layout.
- Write `ontology.py`, the toy fixtures and `test_ontology.py` (including the xref-exclusion test).
- Implement `cli stats`.
- Report to the user, and stop for a decision on the splits:
  - the provided splits and candidate files (Section 3.2);
  - whether definitions exist;
  - cross-reference counts by namespace;
  - disjointness axioms;
  - the % of single-label classes.
- ✅ Tests pass. `results/data_stats.json` exists. The annotation-property counts are printed and the defaults confirmed.

**Phase 1: evaluation harness (≈1.5h)**
- Create the splits (Section 3.2, as decided) and write `evaluate.py`.
- Write `test_evaluate.py` with hand-computed expected values.
- ✅ Tests pass exactly, including null-reference handling.

**Phase 2: lexicon, pair mining and batching (≈2.5h)**
- Write `lexicon.py` (abbreviations, rewrite rules, normalisation table) and `test_lexicon.py`:
  - Schwartz–Hearst on known examples;
  - rule mining and application on toy labels;
  - the variant filters.
- Write `pairs.py` (all positive and negative sources, soft-negative fill), `batching.py`, and the related tests.
- Implement `cli mine`.
- ✅ The batching tests prove all three batch rules. The pair stats (including lexicon contributions and single-label coverage) are printed for each positive and negative configuration. The leakage assert passes.

### Day 2

**Phase 3: training (≈1.5h)**
- Write `train.py` with MNRL, symmetric MNRL and cached MNRL (SupCon if time allows, E3c), plus run-dir logging.
- ✅ `smoke.yaml` trains end to end. The loss decreases. The checkpoint reloads.

**Phase 4: representation and matching (≈2.5h)**
- Write `represent.py` and `match.py`:
  - all class representations, scorings (cosine, CSLS) and selections;
  - neighbour-aware re-scoring;
  - the string-match baseline;
  - the chunked scoring.
- Write `test_match.py`: Stable Marriage, mutual nearest neighbours and CSLS on toy matrices.
- ✅ `cli run --config configs/smoke.yaml` produces metrics. The baseline and E0 (epoch-0) produce full-task metrics.

**Phase 5: experiments (≈3h, mostly compute)**
- Write the configs for B, E0–E5b and S, and run them in priority order. "Should"/"could" rows only if time allows.
- ✅ `results/summary.csv` has every row. The final configuration (chosen on validation) is identified and noted in `results/final_config.md`.

### Day 3

**Phase 6: analysis (≈2h)**
- Write `analysis.py`: error categories, examples, lexicon examples, UMAP and plots (Section 7).
- ✅ All figures, `error_examples.md` and `lexicon_examples.md` exist.

**Phase 7: deliverables (≈1.5h)**
- `notebooks/results.ipynb`: loads `summary.csv` and shows:
  - the main table (final config with 3 seeds and CI, plus the baseline and E0);
  - the ablation tables;
  - all figures, error examples and lexicon examples;
  - each with a short markdown interpretation.
- `README.md`: the problem, the method (with a pipeline diagram in ASCII or mermaid), setup, the data download, how to reproduce each experiment, CLI demo commands and the results summary.
- `results/slides_material.md`: the bullet points, tables and figure paths for the RR2 presentation (about 10 minutes, about 5 content slides).
- ✅ A fresh clone can follow the README to reproduce the smoke run, and the notebook runs top to bottom.

**Phase 8: presentation (≈1.5h)**
- Build the RR2 presentation from `results/slides_material.md` in `presentation/`. The format (e.g. LaTeX/Beamer or a short LaTeX report) is decided with the user at the start of this phase.
- ✅ The presentation compiles/opens and covers the problem, method, main results, ablations and error analysis in about 5 content slides.

---

## 10. Extensions (only after Phase 8, in this order)

1. **Mapping extension:** BERTMap-style iterative extension. For each matched (s, t), score the pairs of their parents and of their children and add those above a threshold. Measure the ΔP/R/F1 over the core result.
2. **Consistency filter / repair:**
   - (a) A lightweight structural filter: if A≡A′ and B≡B′ with A ⊑ B but B′ ⊑ A′ (crossed hierarchy), or if A′ and B′ are disjoint, drop the lower-scoring mapping.
   - (b) Optionally, the full repair via DeepOnto (needs Java).
   - Measure the Δ.
3. **Repair-guided retraining (self-training):** mappings removed by the filter become cross-ontology hard negatives, and confident mutual-NN mappings that pass it become cross-ontology positives. Retrain once and measure the gain and any drift. Validation is still used only for θ.
4. **Second dataset:** OAEI Conference (Section 3.4).
   - Add the `struct` positive source: anchor = a label of C; positive = a structural view built **without the anchor's own label**: `"subclass of <p1>, <p2>; has subclasses <c1>, <c2>, <c3>"`, using the first label of each parent/child, capped by `max_context_items` (default 3), with `max_seq_length` 64. Also add `class_repr: struct`.
   - Compare `positives: [syn, struct]` vs `[syn]`. Report micro-averaged P/R/F1 over the 21 tasks.
   - Use the same protocol, but choose θ on 3 fixed held-out tasks and report the other 18.
5. **PubMedBERT (E6)**, if it was not done earlier.

The core results table must not change because of extensions. Report extensions in a separate table.

---

## 11. Conventions

- Every function that touches data takes the config object. No hard-coded paths outside `configs/`.
- Fixed seeds: `random`, `numpy`, `torch` and the dataset shuffling.
- Use `logging` instead of print in library code. The CLI prints a short summary.
- Nothing under `data/raw`, `data/pairs` or `runs/` is committed. `results/` is committed.
- Keep code simple and readable. This is a course project to be presented.
