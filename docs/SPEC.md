# SPEC — Contrastive Ontology Matching (ACL course, RR2)

Implementation spec for a Claude Code session. Work **phase by phase** (Section 9). Do not start a phase until the previous phase's acceptance checks pass.

Deadline: **3 days**. The core (Phases 0–7) must be finished first. Extensions (Section 10) only if time remains.

---

## 1. Goal

**Research question.** Can a bi-encoder trained with a **contrastive (InfoNCE) objective** on positive/negative pairs mined from each ontology produce a reusable **entity embedding space** in which ontology matching works by **nearest-neighbour search alone**, without candidate selection, rule-based mapping extension or logical repair?

**Context (from RR1).**
- DeepAlignment (Kolyvakis et al., 2018) refines static word vectors with synonym attract / antonym repel and matches directly in that space (set-to-set cosine distance, then Stable Marriage).
- BERTMap (He et al., 2022) mines synonym/non-synonym label pairs from the ontologies and fine-tunes a cross-encoder classifier. Because it is a cross-encoder, it needs a sub-word candidate index and adds rule-based extension and logic-based repair.
- This project keeps BERTMap-style self-mined pairs, replaces the classifier with a contrastive bi-encoder, and matches in the embedding space the way DeepAlignment does.

**Hard constraints.**
- **Unsupervised setting:** no reference mappings are ever used for training.
- **Test references are never used for any decision.** That includes thresholds, hyperparameters and early stopping.
- **No rule-based or logic-based post-processing in the core.** That kind of processing is only allowed in Extensions.

---

## 2. Environment

- **Hardware:** Mac mini M4, PyTorch **MPS** backend (`device="mps"`). Fallback: Google Colab T4 (`device="cuda"`). The device is set in the config, with an `auto` mode: cuda > mps > cpu.
- **MPS notes:**
  - Set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
  - **No fp16/bf16 mixed precision on MPS**: train in fp32.
  - Keep `max_seq_length` small (labels are short; default 32, 64 for structural views).
- **Python:** 3.11, in a venv (`python -m venv .venv`).
- **Core dependencies:** `torch`, `sentence-transformers>=3`, `datasets`, `owlready2` and/or `rdflib`, `numpy`, `pandas`, `scipy`, `pyyaml`, `tqdm`, `pytest`, `umap-learn`, `matplotlib`, `jupyter`.
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

- **Validation split:** if the package provides an unsupervised validation split, use it. Otherwise create a fixed 10% validation split of the full references (seed 42) and save it to `data/splits/val.tsv` and `data/splits/test.tsv`.
- **Null references:** when scoring on test, predictions involving validation mappings are ignored, and vice versa.
- **Ranking candidates:** `test.cands.tsv` rows whose reference mapping falls in validation are moved out of the test ranking set. If the provided candidates already align with the provided splits, keep them as given.

### 3.3 Leakage rules

1. **Training pairs come only from the internal content of each ontology:** labels, synonyms and hierarchy. Never from mappings.
2. **No encoder pre-trained on UMLS synonym pairs** (e.g. SapBERT), because the Bio-ML ontologies overlap with UMLS and MONDO.
3. **The pair miner asserts that no (source-class, target-class) reference pair appears in its output.** This is trivially true when mining per ontology, but keep the assert.

### 3.4 Optional second dataset (Extension 10.4): OAEI Conference

- **What it is:** 16 small conference-organisation ontologies (OntoFarm). The 21 reference alignments cover the 7 that have gold mappings. Each ontology has about 14–140 classes and usually one label per class (camelCase local names).
- **Source:** https://oaei.ontologymatching.org/2025/conference/. Use `rar2` as the main reference.
- **Why include it:** it is label-poor, so synonym positives barely exist, and it is DeepAlignment's benchmark. It tests the structural-view positives.

---

## 4. Method

### 4.1 Ontology parsing (`src/contrastive_om/ontology.py`)

For each named class, collect:
- `iri`
- `labels`: a deduplicated list of normalised strings. Lowercase, split camelCase and underscores, collapse whitespace.
- `parents` and `children`: direct, named classes only. Drop `owl:Thing` and anonymous restrictions.
- `siblings`: classes sharing at least one direct parent.
- `ancestors`: transitive, cached.
- `disjoint_with`, if present.

**Label annotation properties** are configurable. Default set: `rdfs:label`, `skos:prefLabel`, `skos:altLabel`, `oboInOwl:hasExactSynonym`. Optional: `oboInOwl:hasRelatedSynonym`, `hasNarrowSynonym`, `hasBroadSynonym`, which are noisy and off by default.

In Phase 0, print which annotation properties exist and how often.

**Stats to report per ontology:**
- number of classes;
- labels per class (mean, median, % with ≥2 labels);
- depth;
- siblings per class;
- number of disjointness axioms.

Save to `results/data_stats.json`.

Skip obsolete or deprecated classes (`owl:deprecated true`).

### 4.2 Pair mining (`src/contrastive_om/pairs.py`)

Mine **per ontology**, independently. Output rows look like `{anchor, positive, negatives:[...], class_iri, ontology}` and are saved to `data/pairs/<config-hash>.jsonl`. Also save a stats file with pair counts, the % of classes covered, and the negatives per anchor.

**Positive sources** (config `positives`, list):

| key | anchor | positive | notes |
|---|---|---|---|
| `syn` | label *a* of class C | another label *b* of C | all ordered pairs, capped at `max_pairs_per_class` (default 10) |
| `struct` | a label of C | structural view of C | view = `"<label>, subclass of <p1>, <p2>; has subclasses <c1>, <c2>, <c3>"`, using the first label of each parent/child and capped by `max_context_items` (default 3); gives positives to classes with one label |

**Hard-negative sources** (config `hard_negatives`, list; in-batch negatives always apply):

| key | negative | notes |
|---|---|---|
| `sibling` | a label of a sibling class | the hardest; can be a false negative |
| `sibling_filtered` | as `sibling`, after filtering | drop a sibling if (a) it shares any normalised label with C, (b) its pre-trained-encoder cosine to the anchor is > `filter_cos` (default 0.85), or (c) disjointness axioms exist and do not declare it disjoint (only if the ontology has any disjointness axioms) |
| `hierarchy` | a label of a parent or child of C | related but not equivalent |

`num_hard_negatives` (default 1) is the number of hard negatives per row. A row with no available hard negative falls back to in-batch only, and the miner counts these rows.

**Batch-composition rules** are critical for InfoNCE correctness:
- **Class-unique batches.** A batch never contains two rows with the same `class_iri`. Otherwise one class's synonyms become negatives for each other.
- **Per-ontology batches** (`batch_scope: per_ontology`, the default). All rows in a batch come from one ontology. Otherwise a class and its true cross-ontology match can meet in a batch as a false negative.
- **Implementation:** either pre-build batches and disable trainer shuffling, or override the trainer's batch sampler. A unit test must verify both rules.

### 4.3 Contrastive objective and training (`src/contrastive_om/train.py`)

- **Model:** `SentenceTransformer`. Default encoder `sentence-transformers/all-MiniLM-L6-v2`: 6 layers, hidden size 384, about 22M parameters, mean pooling, already contrastively pre-trained on sentence pairs.
- **Optional encoder (E6):** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (PubMedBERT): BERT-base, 110M parameters, biomedical vocabulary, no sentence-level pretraining. Mean pooling is added automatically.
- **Loss:** InfoNCE via `losses.MultipleNegativesRankingLoss(model, scale=1/τ)`. For anchor *a_i*, positive *p_i* and hard negatives *n_ik*, the negatives are all other positives in the batch plus all hard negatives in the batch.
  - Config `loss: mnrl | mnrl_symmetric | cached_mnrl`.
  - Use `cached_mnrl` (`CachedMultipleNegativesRankingLoss`) for batch sizes above what fits in memory.
- **Defaults:** τ = 0.05 (scale 20), batch size 128, 1 epoch, lr 2e-5, warmup 10%, AdamW, weight decay 0.01, `max_seq_length` 32 (64 if `struct` is used), seed 42.
- **Training data:** pairs from **both** ontologies of the task. This is unsupervised: no mappings are used.
- **Per-run output:** `runs/<YYYYmmdd-HHMM>-<config-name>/`, containing the config copy, `model/`, `train_log.jsonl` (loss per step), `metrics.json` and `timing.json`.

### 4.4 Class representation (`src/contrastive_om/represent.py`)

Config `class_repr`:
- `max` (default): embed every label. The score of a class pair is the maximum cosine over all their label pairs.
- `mean`: the class embedding is the L2-normalised mean of its label embeddings.
- `struct`: the class embedding is the embedding of its structural-view string.

Embeddings are L2-normalised, so cosine is the dot product.

### 4.5 Matching from the embedding space (`src/contrastive_om/match.py`)

1. Encode all classes of the source and the target ontology once, with batched encoding cached to `.npy` in the run dir.
2. Compute source × target scores in chunks. Never materialise the full label × label matrix at once; for `max`, reduce label scores to class scores chunk by chunk.
3. Keep the top-k targets per source (k = 10) and their scores.
4. Selection (config `selection`):
   - `argmax`: the best target per source.
   - `mutual`: keep (s, t) only if t is the best target for s **and** s is the best source for t.
   - `stable_marriage`: one-to-one Gale–Shapley over the top-k lists.
5. Threshold: keep a mapping if score ≥ θ. θ is chosen on the **validation** split (Section 5.2).
6. Output: `mappings.tsv` (`SrcEntity`, `TgtEntity`, `Score`).

No candidate index, no extension, no repair.

---

## 5. Evaluation (`src/contrastive_om/evaluate.py`)

### 5.1 Metrics

| Group | Metric | Computed on |
|---|---|---|
| Global matching | Precision, Recall, F1 | test refs; validation refs as null references (ignored) |
| Local ranking | MRR, Hits@1, Hits@5 | test candidate lists (each gold target ranked among its 100 candidates) |
| Full retrieval | Recall@1/5/10, MRR | each test ref's gold target ranked among **all** target classes |
| Embedding space | alignment, uniformity (Wang & Isola, 2020) | alignment = E‖f(x)−f(y)‖² over held-out synonym pairs; uniformity = log E exp(−2‖f(x)−f(y)‖²) over 5k random labels |
| Cost | training time, encoding time, matching time, peak memory if available | per run |

### 5.2 Protocol

- **Threshold:** sweep θ from 0.50 to 1.00 in steps of 0.01 on **validation**, pick the θ with the best validation F1, then report test P/R/F1 at that θ. Also save the full validation P/R/F1-vs-θ curve.
- **Reference point:** the pre-trained encoder with no fine-tuning ("epoch 0"), evaluated with the same pipeline. This shows the effect of contrastive training.
- **Robustness:** the final configuration is trained with 3 seeds (42, 43, 44) and reported as mean ± std. A bootstrap 95% CI (1,000 resamples of test mappings) is computed for its F1 and MRR.
- **Results table:** all runs append one row to `results/summary.csv`, with config name, key config fields, all metrics, θ and times.

### 5.3 Harness validation

`tests/test_evaluate.py` contains hand-made predictions and references with known P/R/F1, MRR and Hits@k, including the null-reference behaviour. All must match exactly.

---

## 6. Experiments

Each experiment is one YAML in `configs/`, run with `python -m contrastive_om.cli run --config configs/<name>.yaml`. The best setting from each experiment becomes the default for the next.

| ID | Varies | Values | Training runs | Priority |
|---|---|---|---|---|
| E0 | epoch-0 reference | pre-trained MiniLM | 0 | must |
| E1 | positives | `[syn]` / `[syn, struct]` | 2 | must |
| E2 | hard negatives | none / `sibling` / `sibling_filtered` / `hierarchy` | 3 (none = E1 winner) | must |
| E3a | temperature | τ ∈ {0.02, 0.05, 0.1} | 2 (0.05 = E2 winner) | must |
| E3b | batch size | {32, 128, 512} (`cached_mnrl` for 512) | 2 | should |
| E4 | class representation | `max` / `mean` / `struct` | 0 (inference only) | must |
| E5 | selection | `argmax` / `mutual` / `stable_marriage` | 0 (inference only) | must |
| E6 | encoder | PubMedBERT on the final config | 1 | could |
| S | seeds | final config × 3 seeds | 2 | must |

The total is about 10–12 short training runs. MiniLM runs should take minutes on the M4. If a single run exceeds about 20 minutes, reduce `max_pairs_per_class` or switch that run to Colab.

---

## 7. Analysis (Phase 6)

1. **Automatic error categories.**
   - For each test false positive (s, t_pred) with gold target t_gold, classify how t_pred relates to t_gold in the target ontology: `sibling`, `parent`, `child`, `ancestor/descendant (≤3 hops)`, `unrelated`. Also note whether the labels have high token overlap (Jaccard ≥ 0.5).
   - For each false negative, record the rank of t_gold and whether the source or target has only one label.
   - Report category counts for E0 vs the final model, and for E2 `none` vs `sibling_filtered`.
2. **Qualitative examples:** 10 FP and 10 FN examples with labels and scores, in `results/error_examples.md`.
3. **Embedding figure:** a UMAP of the class embeddings of one disease subtree, plus its gold-matched classes from the other ontology. Show it for E0 and for the final model, with lines connecting gold pairs. Save as `results/figures/umap_*.png`.
4. **Plots:** the validation P/R/F1-vs-θ curve, ablation bar charts for E1–E5, and a τ sweep plot. Save to `results/figures/`.

---

## 8. Repository layout

```
contrastive-ontology-matching/
  SPEC.md  CLAUDE.md  README.md  pyproject.toml  .gitignore
  configs/            # base.yaml + one yaml per experiment row; smoke.yaml
  data/README.md      # actual dataset layout (Phase 0)
  data/raw/ data/splits/ data/pairs/      # git-ignored except README
  scripts/download_data.sh
  src/contrastive_om/
    __init__.py config.py ontology.py pairs.py batching.py
    train.py represent.py match.py evaluate.py analysis.py cli.py
  tests/fixtures/toy_a.owl toy_b.owl toy_refs.tsv
  tests/test_ontology.py test_pairs.py test_batching.py test_evaluate.py test_match.py
  notebooks/results.ipynb
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
pairs:
  positives: [syn]
  hard_negatives: []        # sibling | sibling_filtered | hierarchy
  num_hard_negatives: 1
  max_pairs_per_class: 10
  max_context_items: 3
  filter_cos: 0.85
  batch_scope: per_ontology
model:
  encoder: sentence-transformers/all-MiniLM-L6-v2
  max_seq_length: 32
train:
  enabled: true             # false => epoch-0 reference
  loss: mnrl                # mnrl | mnrl_symmetric | cached_mnrl
  temperature: 0.05
  batch_size: 128
  epochs: 1
  lr: 2.0e-5
  warmup_ratio: 0.1
match:
  class_repr: max           # max | mean | struct
  selection: argmax         # argmax | mutual | stable_marriage
  top_k: 10
  threshold_grid: [0.50, 1.00, 0.01]
```

`configs/smoke.yaml` subsamples about 500 classes per ontology and about 2k pairs, for end-to-end runs under 2 minutes.

### CLI (`python -m contrastive_om.cli <cmd>`)

- `stats --config C`: parse the ontologies and write data stats.
- `mine --config C`: mine pairs and print pair stats.
- `train --config C`: train and save the model to a run dir.
- `match --config C --model PATH [--source X.owl --target Y.owl --threshold T --out mappings.tsv]`: works on **any** two OWL files (this is the demo).
- `evaluate --config C --run RUN_DIR`: compute all metrics and append to `summary.csv`.
- `run --config C`: mine → train → match → evaluate, end to end.
- `inspect --model PATH --source X.owl --target Y.owl --query "label"`: print the top-k target classes for a label (demo aid).

---

## 9. Phases

Each phase ends with its acceptance checks passing and a git commit `phase N: <summary>`. The time budget assumes 3 days.

### Day 1

**Phase 0: setup and data (≈1.5h)**
- Create `pyproject.toml` and the venv, install dependencies, and confirm `torch.backends.mps.is_available()`.
- Write `scripts/download_data.sh`, download and extract OMIM–ORDO, and write `data/README.md` with the real layout.
- Write `ontology.py`, the toy fixtures and `test_ontology.py`.
- Implement `cli stats`.
- ✅ Tests pass. `results/data_stats.json` exists. The annotation-property counts are printed and the defaults confirmed.

**Phase 1: evaluation harness (≈1.5h)**
- Create the splits (Section 3.2) and write `evaluate.py`.
- Write `test_evaluate.py` with hand-computed expected values.
- ✅ Tests pass exactly, including null-reference handling.

**Phase 2: pair mining and batching (≈2h)**
- Write `pairs.py` (all positive and negative sources), `batching.py`, and the related tests.
- Implement `cli mine`.
- ✅ The batching tests prove both batch rules. The pair stats are printed for each positive and negative configuration. The leakage assert passes.

### Day 2

**Phase 3: training (≈1.5h)**
- Write `train.py` with MNRL, symmetric MNRL and cached MNRL, plus run-dir logging.
- ✅ `smoke.yaml` trains end to end. The loss decreases. The checkpoint reloads.

**Phase 4: representation and matching (≈2h)**
- Write `represent.py` and `match.py` (all class representations and selections), the chunked scoring, and `test_match.py` (Stable Marriage and mutual nearest neighbours on toy matrices).
- ✅ `cli run --config configs/smoke.yaml` produces metrics. E0 (epoch-0) produces full-task metrics.

**Phase 5: experiments (≈3h, mostly compute)**
- Write the configs for E1–E5 and S, and run them in priority order. E3b and E6 only if time allows.
- ✅ `results/summary.csv` has every row. The final configuration is identified and noted in `results/final_config.md`.

### Day 3

**Phase 6: analysis (≈2h)**
- Write `analysis.py`: error categories, examples, UMAP and plots (Section 7).
- ✅ All figures and `error_examples.md` exist.

**Phase 7: deliverables (≈1.5h)**
- `notebooks/results.ipynb`: loads `summary.csv` and shows the main table (final config with 3 seeds and CI, plus E0), the ablation tables, all figures and error examples, each with a short markdown interpretation.
- `README.md`: the problem, the method (with a pipeline diagram in ASCII or mermaid), setup, the data download, how to reproduce each experiment, CLI demo commands and the results summary.
- `results/slides_material.md`: the bullet points, tables and figure paths for the RR2 presentation (about 10 minutes, about 5 content slides).
- ✅ A fresh clone can follow the README to reproduce the smoke run, and the notebook runs top to bottom.

---

## 10. Extensions (only after Phase 7, in this order)

1. **Mapping extension:** BERTMap-style iterative extension. For each matched (s, t), score the pairs of their parents and of their children and add those above a threshold. Measure the ΔP/R/F1 over the core result.
2. **Consistency filter / repair:**
   - (a) A lightweight structural filter: if A≡A′ and B≡B′ with A ⊑ B but B′ ⊑ A′ (crossed hierarchy), or if A′ and B′ are disjoint, drop the lower-scoring mapping.
   - (b) Optionally, the full repair via DeepOnto (needs Java).
   - Measure the Δ.
3. **Repair-guided retraining:** mappings removed by the filter become cross-ontology hard negatives, and confident mutual-NN mappings that pass it become cross-ontology positives. Retrain once and measure the gain and any drift. Validation is still used only for θ.
4. **Second dataset:** OAEI Conference (Section 3.4), with `positives: [syn, struct]` vs `[syn]`. Report micro-averaged P/R/F1 over the 21 tasks. Use the same protocol, but choose θ on 3 fixed held-out tasks and report the other 18.
5. **PubMedBERT (E6)**, if it was not done earlier.

The core results table must not change because of extensions. Report extensions in a separate table.

---

## 11. Conventions

- Every function that touches data takes the config object. No hard-coded paths outside `configs/`.
- Fixed seeds: `random`, `numpy`, `torch` and the dataset shuffling.
- Use `logging` instead of print in library code. The CLI prints a short summary.
- Nothing under `data/raw`, `data/pairs` or `runs/` is committed. `results/` is committed.
- Keep code simple and readable. This is a course project to be presented.
